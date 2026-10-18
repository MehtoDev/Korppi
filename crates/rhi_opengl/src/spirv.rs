//! SPIR-V input for the OpenGL backend: translation to GLSL 4.50 with `naga` (findings RHI-08,
//! RGL-12).
//!
//! Shaders are written once (Vulkan GLSL, HLSL or anything else that compiles to SPIR-V) and
//! both backends consume the same SPIR-V. This backend parses it with naga's SPIR-V front end and
//! writes GLSL 4.50 core, which it then compiles like hand-written GLSL. Cross-compilation was
//! chosen over `GL_ARB_gl_spirv` because it works on every GL 4.5 driver (`ARB_gl_spirv` is core
//! only in 4.6 and has a history of driver bugs) and needs no second code path.
//!
//! The translation adapts three things:
//! - **Coordinates**: none. RHI shaders write +Y-up clip coordinates with depth in [0, 1] on
//!   both backends (ADR-013): Vulkan flips with a negative viewport height and OpenGL uses
//!   `glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE)`. naga's own Y flip and depth remap are
//!   disabled.
//! - **Push constants** become the std140 uniform block at [`crate::PUSH_CONSTANT_BINDING`] that
//!   the backend fills from `push_constants` (RGL-04). naga's GLSL ignores struct member offsets,
//!   so members a stage declares at a later offset (`layout(offset = 16)`, for a stage that only
//!   reads its own part of the range) are preceded by padding ([`pad_to_offsets`]).
//! - **Descriptor bindings** `(set, binding)` map to the GL binding points the pipeline's bind
//!   group layouts give them (RHI-07; `crate::binding`). SPIR-V is therefore parsed and validated
//!   when the shader is created ([`parse_spirv`]) and written as GLSL when a pipeline using it is
//!   created ([`SpirvShader::to_glsl`]).
//! - **Write-only storage buffers**, which naga rejects, are declared read-write.
//! - **Combined image samplers** (`uniform sampler2D` in Vulkan GLSL), which naga's SPIR-V front
//!   end does not accept, are split into an image and a sampler variable first
//!   ([`split_combined_image_samplers`]); naga's GLSL writer combines them again.

use crate::device::PUSH_CONSTANT_BINDING;
use korppi_rhi::{RhiError, ShaderStage, MAX_BINDINGS_PER_GROUP};
use naga::back::glsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{AddressSpace, ResourceBinding};
use std::borrow::Cow;
use std::collections::HashMap;

/// Descriptor set the push-constant block is moved to before translation (one past the last
/// RHI set, so it cannot collide with a real binding).
const PUSH_CONSTANT_SET: u32 = korppi_rhi::MAX_BIND_GROUPS as u32;

/// The kind of resource a shader declares at a binding; each has its own GL binding namespace.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GlResourceKind {
    /// Uniform block.
    UniformBuffer,
    /// Shader storage block.
    StorageBuffer,
    /// Sampled texture (texture unit).
    SampledTexture,
    /// Storage image (image unit).
    StorageTexture,
}

/// A SPIR-V shader stage parsed and validated by naga, written as GLSL once the binding points of
/// a pipeline are known.
pub struct SpirvShader {
    module: naga::Module,
    info: naga::valid::ModuleInfo,
    stage: ShaderStage,
    entry_point: String,
}

fn naga_stage(stage: ShaderStage) -> naga::ShaderStage {
    match stage {
        ShaderStage::Vertex => naga::ShaderStage::Vertex,
        ShaderStage::Fragment => naga::ShaderStage::Fragment,
        ShaderStage::Compute => naga::ShaderStage::Compute,
    }
}

fn failed(what: &str, detail: String) -> RhiError {
    RhiError::ShaderCompilationFailed(format!("SPIR-V to GLSL ({what}): {detail}"))
}

/// Parses and validates the `entry_point` of `stage` in a SPIR-V module given as bytes, in
/// either byte order.
pub fn parse_spirv(
    spirv: &[u8],
    stage: ShaderStage,
    entry_point: &str,
) -> Result<SpirvShader, RhiError> {
    let words = korppi_rhi::spirv_words_from_bytes(spirv)?;
    parse_spirv_words(&words, stage, entry_point)
}

/// Parses and validates the `entry_point` of `stage` in a SPIR-V module of native-order words.
pub fn parse_spirv_words(
    words: &[u32],
    stage: ShaderStage,
    entry_point: &str,
) -> Result<SpirvShader, RhiError> {
    let words = split_combined_image_samplers(words).map_err(|e| failed("parse", e))?;
    let mut module = naga::front::spv::Frontend::new(
        words.iter().copied(),
        &naga::front::spv::Options {
            adjust_coordinate_space: false,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        },
    )
    .parse()
    .map_err(|e| failed("parse", e.to_string()))?;
    if !module
        .entry_points
        .iter()
        .any(|ep| ep.name == entry_point && ep.stage == naga_stage(stage))
    {
        return Err(failed(
            "parse",
            format!("no {stage:?} entry point named `{entry_point}`"),
        ));
    }

    // Push constants become an ordinary uniform block at the backend's push-constant binding.
    let mut push_constants = None;
    for (handle, global) in module.global_variables.iter_mut() {
        // naga accepts write-only access only for storage images; a `writeonly` storage buffer
        // is declared read-write instead, which GLSL treats the same for a shader that only
        // writes it.
        if let AddressSpace::Storage { access } = &mut global.space {
            if !access.contains(naga::StorageAccess::LOAD) {
                *access |= naga::StorageAccess::LOAD;
            }
        }
        if global.space == AddressSpace::Immediate {
            global.space = AddressSpace::Uniform;
            global.binding = Some(ResourceBinding {
                group: PUSH_CONSTANT_SET,
                binding: 0,
            });
            push_constants = Some(handle);
        }
    }
    if let Some(global) = push_constants {
        pad_to_offsets(&mut module, global).map_err(|e| failed("push constants", e))?;
    }

    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| failed("validation", format!("{e}: {:?}", e.as_inner())))?;
    Ok(SpirvShader {
        module,
        info,
        stage,
        entry_point: entry_point.to_owned(),
    })
}

impl SpirvShader {
    /// The shader's stage.
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    /// Writes GLSL 4.50 core. `binding_point(set, binding, kind)` gives the GL binding point of
    /// each resource the module declares (or explains why it has none).
    ///
    /// `draw_parameters` enables `GL_ARB_shader_draw_parameters`, so that `gl_InstanceIndex`
    /// includes the base instance as it does on Vulkan.
    pub fn to_glsl(
        &self,
        binding_point: &dyn Fn(u32, u32, GlResourceKind) -> Result<u8, String>,
        draw_parameters: bool,
    ) -> Result<String, RhiError> {
        let module = &self.module;
        let mut binding_map = glsl::BindingMap::default();
        for (_, global) in module.global_variables.iter() {
            let Some(binding) = &global.binding else {
                continue;
            };
            let kind = match (&module.types[global.ty].inner, global.space) {
                // GLSL has no standalone samplers: sampler state comes from the texture unit.
                (naga::TypeInner::Sampler { .. }, _) => continue,
                (_, AddressSpace::Uniform) if binding.group == PUSH_CONSTANT_SET => {
                    binding_map.insert(*binding, PUSH_CONSTANT_BINDING as u8);
                    continue;
                }
                (_, AddressSpace::Uniform) => GlResourceKind::UniformBuffer,
                (_, AddressSpace::Storage { .. }) => GlResourceKind::StorageBuffer,
                (
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Storage { .. },
                        ..
                    },
                    _,
                ) => GlResourceKind::StorageTexture,
                (naga::TypeInner::Image { .. }, _) => GlResourceKind::SampledTexture,
                (other, space) => {
                    return Err(failed(
                        "bindings",
                        format!(
                            "set {} binding {}: unsupported resource {other:?} in {space:?}",
                            binding.group, binding.binding
                        ),
                    ))
                }
            };
            let point = binding_point(binding.group, binding.binding, kind)
                .map_err(|e| failed("bindings", e))?;
            binding_map.insert(*binding, point);
        }

        let options = glsl::Options {
            version: glsl::Version::Desktop(450),
            writer_flags: if draw_parameters {
                glsl::WriterFlags::DRAW_PARAMETERS
            } else {
                glsl::WriterFlags::empty()
            },
            binding_map,
            zero_initialize_workgroup_memory: true,
        };
        let pipeline_options = glsl::PipelineOptions {
            shader_stage: naga_stage(self.stage),
            entry_point: self.entry_point.clone(),
            multiview: None,
        };
        let mut glsl_source = String::new();
        let mut writer = glsl::Writer::new(
            &mut glsl_source,
            module,
            &self.info,
            &options,
            &pipeline_options,
            naga::proc::BoundsCheckPolicies::default(),
        )
        .map_err(|e| failed("GLSL writer", e.to_string()))?;
        writer
            .write()
            .map_err(|e| failed("GLSL writer", e.to_string()))?;
        tracing::trace!(
            target: "korppi_rhi_opengl::spirv",
            "{:?} `{}`:\n{glsl_source}",
            self.stage,
            self.entry_point
        );
        Ok(glsl_source)
    }
}

/// Makes the std140 layout of the struct behind `global` match its declared member offsets by
/// inserting `float` padding members before every member that starts after the previous one
/// ends, and renumbers the member accesses of `global` accordingly.
fn pad_to_offsets(
    module: &mut naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
) -> Result<(), String> {
    let ty = module.global_variables[global].ty;
    let (members, span) = match &module.types[ty].inner {
        naga::TypeInner::Struct { members, span } => (members.clone(), *span),
        _ => return Ok(()),
    };
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|e| format!("layout: {e}"))?;
    let float = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::F32),
        },
        naga::Span::UNDEFINED,
    );
    let mut padded = Vec::with_capacity(members.len());
    let mut new_index = Vec::with_capacity(members.len());
    let mut end = 0;
    for member in &members {
        if member.offset < end || (member.offset - end) % 4 != 0 {
            return Err(format!(
                "member at offset {} cannot be padded from offset {end}",
                member.offset
            ));
        }
        let mut offset = end;
        while offset < member.offset {
            padded.push(naga::StructMember {
                name: Some(format!("_pad{offset}")),
                ty: float,
                binding: None,
                offset,
            });
            offset += 4;
        }
        new_index.push(padded.len() as u32);
        padded.push(member.clone());
        end = member.offset + layouter[member.ty].size;
    }
    if padded.len() == members.len() {
        return Ok(());
    }
    let name = module.types[ty].name.clone();
    let padded_ty = module.types.insert(
        naga::Type {
            name,
            inner: naga::TypeInner::Struct {
                members: padded,
                span,
            },
        },
        naga::Span::UNDEFINED,
    );
    module.global_variables[global].ty = padded_ty;

    // Accesses go through the global's pointer (`pc.member`) or a loaded copy of the block.
    let functions = module
        .functions
        .iter_mut()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter_mut().map(|ep| &mut ep.function));
    for function in functions {
        let mut block_values = std::collections::HashSet::new();
        for (handle, expression) in function.expressions.iter() {
            match *expression {
                naga::Expression::GlobalVariable(g) if g == global => {
                    block_values.insert(handle);
                }
                naga::Expression::Load { pointer } if block_values.contains(&pointer) => {
                    block_values.insert(handle);
                }
                _ => {}
            }
        }
        for (_, expression) in function.expressions.iter_mut() {
            if let naga::Expression::AccessIndex { base, index } = expression {
                if block_values.contains(base) {
                    *index = new_index[*index as usize];
                }
            }
        }
    }
    Ok(())
}

const SPIRV_MAGIC: u32 = 0x0723_0203;
const OP_ENTRY_POINT: u32 = 15;
const OP_TYPE_SAMPLER: u32 = 26;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_POINTER: u32 = 32;
const OP_FUNCTION_PARAMETER: u32 = 55;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_DECORATE: u32 = 71;
const OP_SAMPLED_IMAGE: u32 = 86;
const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
/// Descriptor set of the sampler variables split from combined image samplers. naga requires
/// unique bindings; GLSL has no separate samplers (the backend binds sampler objects to the
/// texture unit), so these bindings never reach the GL program.
const SPLIT_SAMPLER_SET: u32 = PUSH_CONSTANT_SET + 1;
/// SPIR-V 1.4, from which entry points list every global variable they use.
const VERSION_1_4: u32 = 0x0001_0400;

fn instruction(opcode: u32, operands: &[u32]) -> impl Iterator<Item = u32> + '_ {
    std::iter::once(((operands.len() as u32 + 1) << 16) | opcode).chain(operands.iter().copied())
}

/// Splits every combined image-sampler variable (`OpTypeSampledImage` in the `UniformConstant`
/// storage class) into an image variable, which keeps the original ID and decorations, and a new
/// sampler variable bound at `(SPLIT_SAMPLER_SET, set * 16 + binding)`. Each load of the combined variable becomes loads
/// of both plus `OpSampledImage`. Returns the module unchanged if it has no such variable.
///
/// # Errors
/// Arrays of combined samplers and functions taking one as a parameter are not supported.
pub fn split_combined_image_samplers(words: &[u32]) -> Result<Cow<'_, [u32]>, String> {
    if words.len() < 5 || words[0] != SPIRV_MAGIC {
        return Err("not a SPIR-V module (bad magic number)".into());
    }
    let version = words[1];
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err(format!("malformed instruction at word {at}"));
        }
        instructions.push((words[at] & 0xFFFF, &words[at + 1..at + count]));
        at += count;
    }

    // Image type of each sampled-image type, and every UniformConstant pointer type.
    let mut sampled_images: HashMap<u32, u32> = HashMap::new();
    let mut constant_pointers: HashMap<u32, u32> = HashMap::new();
    let mut array_elements: HashMap<u32, u32> = HashMap::new();
    for &(opcode, operands) in &instructions {
        match opcode {
            OP_TYPE_SAMPLED_IMAGE => {
                sampled_images.insert(operands[0], operands[1]);
            }
            OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY => {
                array_elements.insert(operands[0], operands[1]);
            }
            OP_TYPE_POINTER if operands[1] == STORAGE_CLASS_UNIFORM_CONSTANT => {
                constant_pointers.insert(operands[0], operands[2]);
            }
            _ => {}
        }
    }
    let pointee = |pointer: u32| constant_pointers.get(&pointer).copied();
    // Combined variables and the image type of each.
    let mut combined: HashMap<u32, u32> = HashMap::new();
    for &(opcode, operands) in &instructions {
        let pointer_type = match opcode {
            OP_VARIABLE if operands[2] == STORAGE_CLASS_UNIFORM_CONSTANT => operands[0],
            OP_FUNCTION_PARAMETER => operands[0],
            _ => continue,
        };
        let Some(pointee) = pointee(pointer_type) else {
            continue;
        };
        let element = array_elements.get(&pointee).copied();
        if element.is_some_and(|e| sampled_images.contains_key(&e)) {
            return Err(
                "arrays of combined image samplers are not supported on OpenGL; \
                        use separate textures and samplers"
                    .into(),
            );
        }
        if let Some(&image_type) = sampled_images.get(&pointee) {
            if opcode == OP_FUNCTION_PARAMETER {
                return Err(
                    "functions taking a combined image sampler are not supported on \
                            OpenGL; use the global sampler directly"
                        .into(),
                );
            }
            combined.insert(operands[1], image_type);
        }
    }
    if combined.is_empty() {
        return Ok(Cow::Borrowed(words));
    }
    // Original (set, binding) of each combined variable, for its sampler's unique binding.
    let mut sets: HashMap<u32, u32> = HashMap::new();
    let mut bindings: HashMap<u32, u32> = HashMap::new();
    for &(opcode, operands) in &instructions {
        if opcode == OP_DECORATE && combined.contains_key(&operands[0]) && operands.len() > 2 {
            match operands[1] {
                DECORATION_DESCRIPTOR_SET => sets.insert(operands[0], operands[2]),
                DECORATION_BINDING => bindings.insert(operands[0], operands[2]),
                _ => None,
            };
        }
    }

    // New IDs: the sampler type and pointer types (unless declared already), one sampler
    // variable per combined variable, and two loads per rewritten load.
    let mut bound = words[3];
    let mut next_id = || {
        let id = bound;
        bound += 1;
        id
    };
    let sampler_type = next_id();
    let sampler_pointer = next_id();
    let mut image_pointers: HashMap<u32, u32> = HashMap::new();
    for &image_type in combined.values() {
        image_pointers
            .entry(image_type)
            .or_insert_with(&mut next_id);
    }
    let sampler_variables: HashMap<u32, u32> =
        combined.keys().map(|&var| (var, next_id())).collect();
    let mut types_declared = false;

    let mut out: Vec<u32> = Vec::with_capacity(words.len() + 16 * combined.len());
    out.extend_from_slice(&words[..5]);
    for &(opcode, operands) in &instructions {
        match opcode {
            OP_ENTRY_POINT if version >= VERSION_1_4 => {
                let mut extended = operands.to_vec();
                for (var, sampler_var) in &sampler_variables {
                    if operands.contains(var) {
                        extended.push(*sampler_var);
                    }
                }
                out.extend(instruction(opcode, &extended));
            }
            OP_DECORATE if sampler_variables.contains_key(&operands[0]) => {
                out.extend(instruction(opcode, operands));
                let var = operands[0];
                let mut copy = operands.to_vec();
                copy[0] = sampler_variables[&var];
                match operands[1] {
                    DECORATION_DESCRIPTOR_SET => copy[2] = SPLIT_SAMPLER_SET,
                    DECORATION_BINDING => {
                        let set = sets.get(&var).copied().unwrap_or(0);
                        copy[2] = set * MAX_BINDINGS_PER_GROUP + operands[2];
                    }
                    _ => {}
                }
                out.extend(instruction(opcode, &copy));
            }
            OP_VARIABLE if combined.contains_key(&operands[1]) => {
                if !types_declared {
                    // Declared just before the first combined variable, after every type it
                    // uses. A separate OpTypeSampler is fine for naga even if one exists.
                    types_declared = true;
                    out.extend(instruction(OP_TYPE_SAMPLER, &[sampler_type]));
                    out.extend(instruction(
                        OP_TYPE_POINTER,
                        &[
                            sampler_pointer,
                            STORAGE_CLASS_UNIFORM_CONSTANT,
                            sampler_type,
                        ],
                    ));
                    for (&image_type, &pointer) in &image_pointers {
                        out.extend(instruction(
                            OP_TYPE_POINTER,
                            &[pointer, STORAGE_CLASS_UNIFORM_CONSTANT, image_type],
                        ));
                    }
                }
                let var = operands[1];
                let image_pointer = image_pointers[&combined[&var]];
                out.extend(instruction(
                    OP_VARIABLE,
                    &[image_pointer, var, STORAGE_CLASS_UNIFORM_CONSTANT],
                ));
                out.extend(instruction(
                    OP_VARIABLE,
                    &[
                        sampler_pointer,
                        sampler_variables[&var],
                        STORAGE_CLASS_UNIFORM_CONSTANT,
                    ],
                ));
            }
            OP_LOAD if combined.contains_key(&operands[2]) => {
                let (result_type, result, var) = (operands[0], operands[1], operands[2]);
                let image = next_id();
                let sampler = next_id();
                out.extend(instruction(OP_LOAD, &[combined[&var], image, var]));
                out.extend(instruction(
                    OP_LOAD,
                    &[sampler_type, sampler, sampler_variables[&var]],
                ));
                out.extend(instruction(
                    OP_SAMPLED_IMAGE,
                    &[result_type, result, image, sampler],
                ));
            }
            // Everything else, including the names of the combined variables (which stay on the
            // image variables), is copied unchanged.
            _ => out.extend(instruction(opcode, operands)),
        }
    }
    out[3] = bound;
    Ok(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    const COMBINED_COMPUTE: &[u8] = include_bytes!("../tests/fixtures/combined_sampler.comp.spv");
    const COMBINED_IN_FUNCTION: &[u8] =
        include_bytes!("../tests/fixtures/combined_samplers_in_function.frag.spv");

    #[test]
    fn modules_without_combined_samplers_are_unchanged() {
        let words = fixture(COMBINED_COMPUTE);
        let split = split_combined_image_samplers(&words).expect("split");
        let again = split_combined_image_samplers(&split).expect("split again");
        assert!(matches!(again, Cow::Borrowed(_)));
    }

    /// Binding points by kind: (set, binding) of each kind in declaration order of the table.
    fn translate(
        spirv: &[u8],
        stage: ShaderStage,
        table: &[(u32, u32, GlResourceKind, u8)],
    ) -> String {
        let shader = parse_spirv(spirv, stage, "main").expect("parse");
        shader
            .to_glsl(
                &|set, binding, kind| {
                    table
                        .iter()
                        .find(|e| (e.0, e.1, e.2) == (set, binding, kind))
                        .map(|e| e.3)
                        .ok_or_else(|| format!("set {set} binding {binding} {kind:?} unmapped"))
                },
                false,
            )
            .expect("translation")
    }

    #[test]
    fn combined_samplers_translate_to_glsl_samplers_at_their_texture_unit() {
        let glsl = translate(
            COMBINED_COMPUTE,
            ShaderStage::Compute,
            &[
                (1, 0, GlResourceKind::SampledTexture, 3),
                (2, 0, GlResourceKind::StorageBuffer, 0),
            ],
        );
        assert!(
            glsl.contains("layout(binding = 3) uniform sampler2D"),
            "set 1 binding 0 must become texture unit 3:\n{glsl}"
        );
        assert!(glsl.contains("layout(std430, binding = 0)"), "{glsl}");
        assert!(glsl.contains("textureLod("), "{glsl}");
    }

    #[test]
    fn push_constant_members_keep_their_offsets() {
        let glsl = translate(
            include_bytes!("../tests/fixtures/push_constant_offset_16.frag.spv"),
            ShaderStage::Fragment,
            &[],
        );
        // `color` is declared at offset 16, so 16 bytes of padding must precede it.
        let pads = glsl.matches("float _pad").count();
        assert_eq!(pads, 4, "{glsl}");
        assert!(glsl.contains("binding = 64"), "{glsl}");
    }

    #[test]
    fn combined_samplers_used_in_helper_functions_translate() {
        let glsl = translate(
            COMBINED_IN_FUNCTION,
            ShaderStage::Fragment,
            &[
                (1, 2, GlResourceKind::SampledTexture, 2),
                (1, 3, GlResourceKind::SampledTexture, 3),
            ],
        );
        assert!(
            glsl.contains("layout(binding = 2) uniform sampler2D"),
            "{glsl}"
        );
        assert!(
            glsl.contains("layout(binding = 3) uniform sampler2D"),
            "{glsl}"
        );
    }

    #[test]
    fn resources_missing_from_the_pipeline_layout_are_an_error() {
        let shader = parse_spirv(COMBINED_COMPUTE, ShaderStage::Compute, "main").expect("parse");
        let result = shader.to_glsl(&|_, _, _| Err("not in the layout".into()), false);
        assert!(
            matches!(&result, Err(RhiError::ShaderCompilationFailed(m)) if m.contains("not in the layout")),
            "{result:?}"
        );
    }

    #[test]
    fn a_missing_entry_point_is_an_error() {
        let result = parse_spirv(COMBINED_COMPUTE, ShaderStage::Fragment, "main");
        assert!(matches!(result, Err(RhiError::ShaderCompilationFailed(_))));
    }

    #[test]
    fn invalid_spirv_is_a_compilation_error() {
        let result = parse_spirv(&[0, 1, 2, 3], ShaderStage::Vertex, "main");
        assert!(matches!(result, Err(RhiError::ShaderCompilationFailed(_))));
    }
}
