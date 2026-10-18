use crate::error::RhiError;
use crate::handle::{BindGroupLayoutHandle, ShaderHandle, TextureHandle};
use crate::types::{
    BorderColor, BufferUsage, ColorWriteMask, CompareFunction, CullMode, FilterMode, FrontFace,
    LoadOp, MemoryUsage, PolygonMode, PrimitiveTopology, SampleCount, SamplerAddressMode, StoreOp,
    TextureAspect, TextureDimension, TextureFormat, TextureUsage, MAX_PUSH_CONSTANTS_SIZE,
};

/// Parameters for creating a GPU buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferDescriptor {
    /// Total byte size of the buffer.
    pub size: u64,
    /// Bitflags describing how the buffer will be accessed.
    pub usage: BufferUsage,
    /// CPU/GPU memory placement.
    pub memory: MemoryUsage,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

impl BufferDescriptor {
    /// Checks the descriptor for `create_buffer`: the size must be non-zero (Vulkan forbids
    /// empty buffers).
    pub fn validate(&self) -> Result<(), RhiError> {
        if self.size == 0 {
            return Err(RhiError::ValidationError(format!(
                "buffer {:?} has size 0",
                self.label.unwrap_or("(unlabelled)")
            )));
        }
        Ok(())
    }

    /// Checks `initial_data` for `create_buffer`: it may be shorter than the buffer (the
    /// remaining bytes are unspecified) but not longer.
    pub fn validate_initial_data(&self, data: &[u8]) -> Result<(), RhiError> {
        if data.len() as u64 > self.size {
            return Err(RhiError::ValidationError(format!(
                "initial data ({} bytes) is larger than the buffer ({} bytes)",
                data.len(),
                self.size
            )));
        }
        Ok(())
    }
}

/// Parameters for creating a GPU texture (RHI-05).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureDescriptor {
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels (1 for `D1`).
    pub height: u32,
    /// Depth in texels of a `D3` texture; 1 for every other dimension.
    pub depth: u32,
    /// Array layers: 1 for `D1`, `D2` and `D3`, 6 for `Cube`, a multiple of 6 for `CubeArray`,
    /// any count for `D2Array`.
    pub array_layers: u32,
    /// Number of mipmap levels (1 for multisampled textures).
    pub mip_levels: u32,
    /// Multisampling sample count (`D2`/`D2Array` attachments only).
    pub sample_count: SampleCount,
    /// Texel format.
    pub format: TextureFormat,
    /// Texture usage bitflags.
    pub usage: TextureUsage,
    /// Dimensionality, which is also the type of the default view.
    pub dimension: TextureDimension,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

impl TextureDescriptor {
    /// Checks the descriptor's internal consistency (not device limits).
    ///
    /// # Errors
    /// `RhiError::ValidationError` describing the first problem found.
    pub fn validate(&self) -> Result<(), RhiError> {
        let fail = |message: String| Err(RhiError::ValidationError(message));
        if self.width == 0 || self.height == 0 || self.depth == 0 || self.array_layers == 0 {
            return fail(format!(
                "texture extent {}x{}x{} with {} layers has a zero dimension",
                self.width, self.height, self.depth, self.array_layers
            ));
        }
        let largest = self.width.max(self.height).max(self.depth);
        let max_mips = 32 - largest.leading_zeros();
        if self.mip_levels == 0 || self.mip_levels > max_mips {
            return fail(format!(
                "{} mip levels requested; a {}x{}x{} texture has 1..={max_mips}",
                self.mip_levels, self.width, self.height, self.depth
            ));
        }
        match self.dimension {
            TextureDimension::D1
                if self.height != 1 || self.depth != 1 || self.array_layers != 1 =>
            {
                return fail("D1 textures have height, depth and array_layers 1".into())
            }
            TextureDimension::D2 if self.depth != 1 || self.array_layers != 1 => {
                return fail("D2 textures have depth and array_layers 1 (use D2Array)".into())
            }
            TextureDimension::D2Array if self.depth != 1 => {
                return fail("D2Array textures have depth 1".into())
            }
            TextureDimension::D3 if self.array_layers != 1 => {
                return fail("D3 textures have one array layer".into())
            }
            TextureDimension::Cube | TextureDimension::CubeArray => {
                if self.width != self.height || self.depth != 1 {
                    return fail("cube faces are square with depth 1".into());
                }
                let six = self.dimension == TextureDimension::Cube && self.array_layers == 6;
                let multiple = self.dimension == TextureDimension::CubeArray
                    && self.array_layers.is_multiple_of(6);
                if !six && !multiple {
                    return fail(format!(
                        "{:?} needs 6 (Cube) or a multiple of 6 (CubeArray) layers, not {}",
                        self.dimension, self.array_layers
                    ));
                }
            }
            _ => {}
        }
        if self.sample_count != SampleCount::Count1 {
            let ok = matches!(
                self.dimension,
                TextureDimension::D2 | TextureDimension::D2Array
            ) && self.mip_levels == 1
                && !self.format.is_compressed();
            if !ok {
                return fail(
                    "multisampled textures are uncompressed D2/D2Array with one mip level".into(),
                );
            }
        }
        if self.format.is_compressed() && self.dimension == TextureDimension::D3 {
            return fail("block-compressed formats cannot be 3D".into());
        }
        Ok(())
    }

    /// Width, height and depth of mip level `mip`.
    pub fn mip_extent(&self, mip: u32) -> (u32, u32, u32) {
        (
            (self.width >> mip).max(1),
            (self.height >> mip).max(1),
            (self.depth >> mip).max(1),
        )
    }

    /// Size in bytes of one tightly packed mip-0 image of one layer (whole blocks for
    /// block-compressed formats).
    pub fn level0_size_bytes(&self) -> u64 {
        self.format.image_bytes(self.width, self.height)
    }

    /// Size in bytes of mip level `mip` across every layer and slice, tightly packed.
    pub fn level_size_bytes(&self, mip: u32) -> u64 {
        let (width, height, depth) = self.mip_extent(mip);
        self.format.image_bytes(width, height) * u64::from(depth) * u64::from(self.array_layers)
    }

    /// Checks `initial_data` for `create_texture` and returns how many mip levels it holds.
    ///
    /// Initial data is either mip level 0 of every layer and slice, or every mip level, each
    /// level tightly packed with layers (cube faces) in order and slices within a layer in
    /// order, levels in order (the layout KTX2 uses). Shorter data made backends read past the
    /// slice (RGL-10).
    pub fn validate_initial_data(&self, data: &[u8]) -> Result<u32, RhiError> {
        let level0 = self.level_size_bytes(0);
        let all: u64 = (0..self.mip_levels)
            .map(|mip| self.level_size_bytes(mip))
            .sum();
        let len = data.len() as u64;
        if len == level0 {
            Ok(1)
        } else if len == all {
            Ok(self.mip_levels)
        } else {
            Err(RhiError::ValidationError(format!(
                "initial data is {len} bytes; a {}x{}x{} {:?} texture with {} layers takes \
                 {level0} bytes for mip 0 or {all} bytes for all {} mips",
                self.width,
                self.height,
                self.depth,
                self.format,
                self.array_layers,
                self.mip_levels
            )))
        }
    }
}

/// Parameters for creating a view of a texture: a range of its mips and layers, seen as some
/// dimension, through some aspect (RHI-05).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    /// The viewed texture. Destroying it destroys its views.
    pub texture: TextureHandle,
    /// How the view's layers are seen: `D2` for one layer, `D2Array`, `Cube` (6 layers of a
    /// square texture), `CubeArray`, or the texture's own `D1`/`D3`.
    pub dimension: TextureDimension,
    /// First mip level.
    pub base_mip_level: u32,
    /// Number of mip levels.
    pub mip_level_count: u32,
    /// First array layer.
    pub base_array_layer: u32,
    /// Number of array layers.
    pub array_layer_count: u32,
    /// Aspect seen by the view (`DepthOnly`/`StencilOnly` for depth/stencil textures).
    pub aspect: TextureAspect,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

impl TextureViewDescriptor {
    /// Checks the view against the viewed texture (`texture`).
    ///
    /// # Errors
    /// `RhiError::ValidationError` for ranges outside the texture, a dimension the texture
    /// cannot be seen as, or an aspect the format lacks.
    pub fn validate(&self, texture: &crate::validation::TextureInfo) -> Result<(), RhiError> {
        let fail = |message: String| Err(RhiError::ValidationError(message));
        let mips_end = self.base_mip_level.checked_add(self.mip_level_count);
        if self.mip_level_count == 0 || mips_end.is_none_or(|end| end > texture.mip_levels) {
            return fail(format!(
                "mips {}..+{} are outside the texture's {} mips",
                self.base_mip_level, self.mip_level_count, texture.mip_levels
            ));
        }
        let layers_end = self.base_array_layer.checked_add(self.array_layer_count);
        if self.array_layer_count == 0 || layers_end.is_none_or(|end| end > texture.array_layers) {
            return fail(format!(
                "layers {}..+{} are outside the texture's {} layers",
                self.base_array_layer, self.array_layer_count, texture.array_layers
            ));
        }
        let layered = matches!(
            texture.dimension,
            TextureDimension::D2
                | TextureDimension::D2Array
                | TextureDimension::Cube
                | TextureDimension::CubeArray
        );
        let ok = match self.dimension {
            TextureDimension::D1 => texture.dimension == TextureDimension::D1,
            TextureDimension::D3 => texture.dimension == TextureDimension::D3,
            TextureDimension::D2 => layered && self.array_layer_count == 1,
            TextureDimension::D2Array => layered,
            TextureDimension::Cube | TextureDimension::CubeArray => {
                let count_ok = if self.dimension == TextureDimension::Cube {
                    self.array_layer_count == 6
                } else {
                    self.array_layer_count.is_multiple_of(6)
                };
                // Cube views need square layers on a texture created cube-compatible: every
                // square texture with at least 6 layers is.
                layered && count_ok && texture.cube_compatible()
            }
        };
        if !ok {
            return fail(format!(
                "a {:?} texture ({} layers) cannot be viewed as {:?} with {} layers",
                texture.dimension, texture.array_layers, self.dimension, self.array_layer_count
            ));
        }
        if texture.sample_count != SampleCount::Count1
            && !matches!(
                self.dimension,
                TextureDimension::D2 | TextureDimension::D2Array
            )
        {
            return fail("multisampled textures can only be viewed as D2/D2Array".into());
        }
        let aspect_ok = match self.aspect {
            TextureAspect::All => true,
            TextureAspect::DepthOnly => texture.format.is_depth(),
            TextureAspect::StencilOnly => texture.format.is_stencil(),
        };
        if !aspect_ok {
            return fail(format!(
                "{:?} has no {:?} aspect",
                texture.format, self.aspect
            ));
        }
        Ok(())
    }
}

/// Parameters for creating a texture sampler.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerDescriptor {
    /// Minification filter.
    pub min_filter: FilterMode,
    /// Magnification filter.
    pub mag_filter: FilterMode,
    /// Mipmap level filtering.
    pub mipmap_filter: FilterMode,
    /// U coordinate wrap mode.
    pub address_mode_u: SamplerAddressMode,
    /// V coordinate wrap mode.
    pub address_mode_v: SamplerAddressMode,
    /// W coordinate wrap mode.
    pub address_mode_w: SamplerAddressMode,
    /// Depth comparison function (if used as a shadow sampler).
    pub compare: Option<CompareFunction>,
    /// Maximum anisotropic filtering level (1 to 16); clamped to the device limit, and ignored
    /// where the device does not support anisotropic filtering.
    pub max_anisotropy: u8,
    /// Border colour for `SamplerAddressMode::ClampToBorder`.
    pub border_color: BorderColor,
    /// Lowest mip level sampled (LOD clamp, applied after the bias).
    pub lod_min_clamp: f32,
    /// Highest mip level sampled; [`SamplerDescriptor::LOD_CLAMP_NONE`] for no clamp.
    pub lod_max_clamp: f32,
    /// Bias added to the computed (or explicit) level of detail; clamped to the device limit
    /// (at least 2 on every device).
    pub mip_lod_bias: f32,
    /// Optional debug label; not serialized (RHI-14).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub label: Option<&'static str>,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: SamplerAddressMode::Repeat,
            address_mode_v: SamplerAddressMode::Repeat,
            address_mode_w: SamplerAddressMode::Repeat,
            compare: None,
            max_anisotropy: 1,
            border_color: BorderColor::TransparentBlack,
            lod_min_clamp: 0.0,
            lod_max_clamp: Self::LOD_CLAMP_NONE,
            mip_lod_bias: 0.0,
            label: None,
        }
    }
}

impl SamplerDescriptor {
    /// `lod_max_clamp` that does not clamp (Vulkan's `VK_LOD_CLAMP_NONE`).
    pub const LOD_CLAMP_NONE: f32 = 1000.0;

    /// Checks the descriptor for `create_sampler`: finite LOD values, the minimum not above the
    /// maximum.
    pub fn validate(&self) -> Result<(), RhiError> {
        let lods = [self.lod_min_clamp, self.lod_max_clamp, self.mip_lod_bias];
        if lods.iter().any(|v| !v.is_finite()) || self.lod_min_clamp > self.lod_max_clamp {
            return Err(RhiError::ValidationError(format!(
                "sampler {:?}: LOD clamp [{}, {}] and bias {} must be finite, minimum <= maximum",
                self.label.unwrap_or("(unlabelled)"),
                self.lod_min_clamp,
                self.lod_max_clamp,
                self.mip_lod_bias
            )));
        }
        Ok(())
    }
}

/// Programmable shader stages.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShaderStage {
    /// Vertex shader.
    Vertex,
    /// Fragment / Pixel shader.
    Fragment,
    /// Compute shader.
    Compute,
}

/// Shader bytecode or source representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderSource<'a> {
    /// SPIR-V 32-bit words (native to Vulkan 1.3).
    Spirv(&'a [u32]),
    /// Raw SPIR-V byte slice.
    SpirvBytes(&'a [u8]),
    /// GLSL 450 Core source text.
    Glsl(&'a str),
}

/// Parameters for creating a shader module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderDescriptor<'a> {
    /// Shader stage.
    pub stage: ShaderStage,
    /// Entry point function name (e.g. "main").
    pub entry_point: &'static str,
    /// Shader bytecode or source text.
    pub source: ShaderSource<'a>,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

/// Data format of a vertex attribute (finding RHI-13 added the half-float and 8/16-bit
/// formats).
///
/// `*Norm`/`Unorm`/`Snorm` formats are read by the shader as normalized floats; `Uint`/`Sint`
/// formats as integer vectors (`uvec`/`ivec`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VertexFormat {
    /// 32-bit float (`f32`, 4 bytes).
    Float32,
    /// 2-component 32-bit float vector (`Vec2`, 8 bytes).
    Float32x2,
    /// 3-component 32-bit float vector (`Vec3`, 12 bytes).
    Float32x3,
    /// 4-component 32-bit float vector (`Vec4`, 16 bytes).
    Float32x4,
    /// 2-component 16-bit float vector (4 bytes): compact UVs.
    Float16x2,
    /// 4-component 16-bit float vector (8 bytes).
    Float16x4,
    /// 32-bit unsigned int (`u32`, 4 bytes).
    Uint32,
    /// 2-component 32-bit uint vector (8 bytes).
    Uint32x2,
    /// 3-component 32-bit uint vector (12 bytes).
    Uint32x3,
    /// 4-component 32-bit uint vector (16 bytes).
    Uint32x4,
    /// 32-bit signed int (4 bytes).
    Sint32,
    /// 2-component 32-bit signed int vector (8 bytes).
    Sint32x2,
    /// 4-component 32-bit signed int vector (16 bytes).
    Sint32x4,
    /// 4-component 8-bit unsigned normalized vector (4 bytes, e.g. packed vertex color).
    Uint8x4Norm,
    /// 4-component 8-bit signed normalized vector (4 bytes): packed normals and tangents.
    Sint8x4Norm,
    /// 4-component 8-bit unsigned int vector (4 bytes): bone indices.
    Uint8x4,
    /// 2-component 16-bit unsigned normalized vector (4 bytes): quantized UVs.
    Uint16x2Norm,
    /// 4-component 16-bit unsigned normalized vector (8 bytes): bone weights.
    Uint16x4Norm,
    /// 2-component 16-bit signed normalized vector (4 bytes).
    Sint16x2Norm,
    /// 4-component 16-bit signed normalized vector (8 bytes): quantized positions/normals.
    Sint16x4Norm,
    /// 2-component 16-bit unsigned int vector (4 bytes).
    Uint16x2,
    /// 4-component 16-bit unsigned int vector (8 bytes).
    Uint16x4,
}

impl VertexFormat {
    /// Returns the byte size of this vertex attribute format.
    #[inline]
    pub const fn size(&self) -> u64 {
        match self {
            Self::Float32
            | Self::Uint32
            | Self::Sint32
            | Self::Uint8x4Norm
            | Self::Sint8x4Norm
            | Self::Uint8x4
            | Self::Float16x2
            | Self::Uint16x2Norm
            | Self::Sint16x2Norm
            | Self::Uint16x2 => 4,
            Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2
            | Self::Float16x4
            | Self::Uint16x4Norm
            | Self::Sint16x4Norm
            | Self::Uint16x4 => 8,
            Self::Float32x3 | Self::Uint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
        }
    }

    /// Returns true if the shader reads this format as an integer vector.
    #[inline]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::Uint32
                | Self::Uint32x2
                | Self::Uint32x3
                | Self::Uint32x4
                | Self::Sint32
                | Self::Sint32x2
                | Self::Sint32x4
                | Self::Uint8x4
                | Self::Uint16x2
                | Self::Uint16x4
        )
    }
}

/// Step mode for stepping through vertex buffer elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VertexStepMode {
    /// Step per vertex.
    Vertex,
    /// Step per instance.
    Instance,
}

/// Layout binding for a vertex buffer slot.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexBinding {
    /// Slot index of the binding.
    pub binding: u32,
    /// Stride in bytes between consecutive elements.
    pub stride: u64,
    /// Advance rate (per vertex or per instance).
    pub step_mode: VertexStepMode,
}

/// Declaration of a single vertex attribute.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexAttribute {
    /// Shader input location (`layout(location = X)`).
    pub location: u32,
    /// Vertex buffer binding index providing this attribute.
    pub binding: u32,
    /// Format and component count.
    pub format: VertexFormat,
    /// Byte offset within the vertex buffer struct.
    pub offset: u64,
}

/// Vertex input layout descriptor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexLayoutDescriptor {
    /// Buffer bindings.
    pub bindings: Vec<VertexBinding>,
    /// Attribute declarations.
    pub attributes: Vec<VertexAttribute>,
}

impl VertexLayoutDescriptor {
    /// Creates an empty vertex layout descriptor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a vertex buffer binding to the layout.
    pub fn with_binding(mut self, binding: VertexBinding) -> Self {
        self.bindings.push(binding);
        self
    }

    /// Adds a vertex attribute to the layout.
    pub fn with_attribute(mut self, attribute: VertexAttribute) -> Self {
        self.attributes.push(attribute);
        self
    }

    /// Validates attribute offsets and binding references.
    pub fn validate(&self) -> Result<(), RhiError> {
        for attr in &self.attributes {
            let binding_exists = self.bindings.iter().any(|b| b.binding == attr.binding);
            if !binding_exists {
                return Err(RhiError::ValidationError(format!(
                    "Vertex attribute at location {} references non-existent binding {}",
                    attr.location, attr.binding
                )));
            }
        }
        Ok(())
    }
}

/// Rasterizer state configuration.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RasterizerState {
    /// Triangle face culling.
    pub cull_mode: CullMode,
    /// Front-face winding orientation.
    pub front_face: FrontFace,
    /// Polygon fill mode.
    pub polygon_mode: PolygonMode,
    /// Constant depth bias added to fragments.
    pub depth_bias: f32,
    /// Slope-scaled depth bias for shadow map anti-acne.
    pub slope_scaled_depth_bias: f32,
    /// Largest (or, if negative, smallest) bias applied; 0 applies the bias unclamped. Non-zero
    /// values need `DeviceFeatures::DEPTH_BIAS_CLAMP`.
    pub depth_bias_clamp: f32,
    /// Clamp fragment depth to the viewport's depth range instead of clipping primitives
    /// against the near and far planes (shadow-map pancaking). Needs
    /// `DeviceFeatures::DEPTH_CLAMP`.
    pub depth_clamp: bool,
}

impl Default for RasterizerState {
    fn default() -> Self {
        Self {
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            polygon_mode: PolygonMode::Fill,
            depth_bias: 0.0,
            slope_scaled_depth_bias: 0.0,
            depth_bias_clamp: 0.0,
            depth_clamp: false,
        }
    }
}

impl RasterizerState {
    /// The device features this state needs.
    pub fn required_features(&self) -> crate::capabilities::DeviceFeatures {
        use crate::capabilities::DeviceFeatures;
        let mut features = DeviceFeatures::empty();
        if self.depth_clamp {
            features |= DeviceFeatures::DEPTH_CLAMP;
        }
        if self.depth_bias_clamp != 0.0 {
            features |= DeviceFeatures::DEPTH_BIAS_CLAMP;
        }
        if self.polygon_mode == PolygonMode::Line {
            features |= DeviceFeatures::POLYGON_MODE_LINE;
        }
        features
    }
}

/// What a stencil test does with the stencil value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StencilOperation {
    /// Keep the value.
    Keep,
    /// Set it to 0.
    Zero,
    /// Set it to the reference (`set_stencil_reference`).
    Replace,
    /// Increment, clamping at the maximum.
    IncrementClamp,
    /// Decrement, clamping at 0.
    DecrementClamp,
    /// Invert the bits.
    Invert,
    /// Increment, wrapping to 0.
    IncrementWrap,
    /// Decrement, wrapping to the maximum.
    DecrementWrap,
}

/// Stencil test of one face: `compare(reference & read_mask, value & read_mask)`, then the
/// operation for the outcome, writing through the write mask.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StencilFaceState {
    /// Comparison of the reference with the stored value.
    pub compare: CompareFunction,
    /// Operation when the stencil test fails.
    pub fail_op: StencilOperation,
    /// Operation when the stencil test passes and the depth test fails.
    pub depth_fail_op: StencilOperation,
    /// Operation when both pass.
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    /// A face that neither tests nor changes the stencil buffer.
    pub const IGNORE: Self = Self {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self::IGNORE
    }
}

/// Depth/Stencil test configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthStencilState {
    /// Format of the depth attachment this pipeline renders with; must equal the format of the
    /// pass's depth texture (Vulkan bakes it into the pipeline).
    pub format: TextureFormat,
    /// Whether depth testing is enabled.
    pub depth_test_enabled: bool,
    /// Whether passing fragments write to the depth buffer.
    pub depth_write_enabled: bool,
    /// Comparison operator for depth test.
    pub depth_compare: CompareFunction,
    /// Stencil test of front-facing primitives.
    pub stencil_front: StencilFaceState,
    /// Stencil test of back-facing primitives.
    pub stencil_back: StencilFaceState,
    /// Bits of the reference and stored value the stencil test compares.
    pub stencil_read_mask: u32,
    /// Bits stencil operations write.
    pub stencil_write_mask: u32,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            format: TextureFormat::Depth32Float,
            depth_test_enabled: true,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual, // Default to Reverse-Z
            stencil_front: StencilFaceState::IGNORE,
            stencil_back: StencilFaceState::IGNORE,
            stencil_read_mask: 0xFF,
            stencil_write_mask: 0xFF,
        }
    }
}

impl DepthStencilState {
    /// True if either face tests or changes the stencil buffer.
    pub fn stencil_enabled(&self) -> bool {
        self.stencil_front != StencilFaceState::IGNORE
            || self.stencil_back != StencilFaceState::IGNORE
    }
}

/// Blending factor.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlendFactor {
    /// Zero.
    Zero,
    /// One.
    One,
    /// Source alpha.
    SrcAlpha,
    /// 1 - Source alpha.
    OneMinusSrcAlpha,
    /// Destination alpha.
    DstAlpha,
    /// 1 - Destination alpha.
    OneMinusDstAlpha,
    /// Source color.
    SrcColor,
    /// 1 - Source color.
    OneMinusSrcColor,
    /// Destination color.
    DstColor,
    /// 1 - Destination color.
    OneMinusDstColor,
    /// The blend constant (`set_blend_constant`).
    Constant,
    /// 1 - the blend constant.
    OneMinusConstant,
}

/// Blending arithmetic operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlendOp {
    /// Add source and destination.
    Add,
    /// Subtract destination from source.
    Subtract,
    /// Subtract source from destination.
    ReverseSubtract,
    /// Take minimum of source and destination.
    Min,
    /// Take maximum of source and destination.
    Max,
}

/// Blend component configuration for color or alpha.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlendComponent {
    /// Factor multiplied with source fragment.
    pub src_factor: BlendFactor,
    /// Factor multiplied with destination buffer.
    pub dst_factor: BlendFactor,
    /// Blending equation.
    pub operation: BlendOp,
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Zero,
            operation: BlendOp::Add,
        }
    }
}

/// Complete color and alpha blend state.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlendState {
    /// Color channel blending.
    pub color: BlendComponent,
    /// Alpha channel blending.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Standard premultiplied alpha blend mode.
    pub const PREMULTIPLIED: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOp::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOp::Add,
        },
    };

    /// Standard non-premultiplied alpha blend mode.
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOp::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOp::Add,
        },
    };
}

/// Render target output configuration for a color attachment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorTargetState {
    /// Target texture format.
    pub format: TextureFormat,
    /// Optional blend state.
    pub blend: Option<BlendState>,
    /// Color channel write mask.
    pub write_mask: ColorWriteMask,
}

/// Push constant range allocation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PushConstantRange {
    /// Shader stages accessing this push constant range.
    pub stages: crate::binding::ShaderStages,
    /// Byte offset within the 128-byte push constant block.
    pub offset: u32,
    /// Byte size of this push constant struct.
    pub size: u32,
}

/// Multisampling state of a graphics pipeline; `count` must equal the sample count of the
/// pass's attachments.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultisampleState {
    /// Samples per pixel.
    pub count: SampleCount,
    /// Derive coverage from fragment alpha (foliage, fences).
    pub alpha_to_coverage: bool,
}

/// Complete descriptor for creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphicsPipelineDescriptor {
    /// Vertex shader module.
    pub vertex_shader: ShaderHandle,
    /// Fragment shader module.
    pub fragment_shader: Option<ShaderHandle>,
    /// Vertex buffer input declarations.
    pub vertex_layout: VertexLayoutDescriptor,
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Rasterizer state.
    pub rasterizer: RasterizerState,
    /// Multisampling state.
    pub multisample: MultisampleState,
    /// Depth/stencil state.
    pub depth_stencil: Option<DepthStencilState>,
    /// Color target formats and blend states (must match attachments in dynamic rendering).
    pub color_targets: Vec<ColorTargetState>,
    /// Layouts of descriptor sets 0, 1, ... (`layout(set = i)` is `bind_group_layouts[i]`).
    pub bind_group_layouts: Vec<BindGroupLayoutHandle>,
    /// Push constant ranges.
    pub push_constant_ranges: Vec<PushConstantRange>,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

impl GraphicsPipelineDescriptor {
    /// Validates that total push constants do not exceed hardware limits.
    pub fn validate(&self) -> Result<(), RhiError> {
        self.vertex_layout.validate()?;
        if let Some(ds) = &self.depth_stencil {
            if ds.stencil_enabled() && !ds.format.is_stencil() {
                return Err(RhiError::ValidationError(format!(
                    "stencil state needs a stencil format; {:?} has no stencil",
                    ds.format
                )));
            }
        }
        if !self.rasterizer.depth_bias_clamp.is_finite() {
            return Err(RhiError::ValidationError(
                "depth_bias_clamp must be finite".into(),
            ));
        }
        for range in &self.push_constant_ranges {
            if (range.offset + range.size) as usize > MAX_PUSH_CONSTANTS_SIZE {
                return Err(RhiError::ValidationError(format!(
                    "Push constant range offset {} + size {} exceeds maximum hardware limit of 128 bytes",
                    range.offset, range.size
                )));
            }
        }
        Ok(())
    }
}

/// Descriptor for creating a compute pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    /// Compute shader stage.
    pub compute_shader: ShaderHandle,
    /// Layouts of descriptor sets 0, 1, ... (`layout(set = i)` is `bind_group_layouts[i]`).
    pub bind_group_layouts: Vec<BindGroupLayoutHandle>,
    /// Push constant ranges.
    pub push_constant_ranges: Vec<PushConstantRange>,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

/// Color attachment target for dynamic rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderingAttachment {
    /// Texture handle to render into.
    pub texture: TextureHandle,
    /// Mipmap level to render into.
    pub mip_level: u32,
    /// Array layer (cube face, or slice of a 3D texture) to render into.
    pub layer: u32,
    /// Load operation at start of pass.
    pub load_op: LoadOp,
    /// Store operation at end of pass.
    pub store_op: StoreOp,
    /// Clear color value if `load_op` is `LoadOp::Clear`.
    pub clear_color: Option<[f32; 4]>,
    /// Single-sample texture (mip 0, layer 0) the multisampled attachment is resolved into at
    /// the end of the pass. It must be in the colour-attachment layout, with the attachment's
    /// format and size.
    pub resolve_target: Option<TextureHandle>,
}

/// Depth/Stencil attachment target for dynamic rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAttachment {
    /// Depth texture handle.
    pub texture: TextureHandle,
    /// Mipmap level to render into.
    pub mip_level: u32,
    /// Array layer (shadow cascade, cube face) to render into.
    pub layer: u32,
    /// Depth load operation.
    pub load_op: LoadOp,
    /// Depth store operation.
    pub store_op: StoreOp,
    /// Depth clear value if `load_op` is `LoadOp::Clear`.
    pub clear_depth: Option<f32>,
    /// Stencil load operation (ignored for depth-only formats).
    pub stencil_load_op: LoadOp,
    /// Stencil store operation (ignored for depth-only formats).
    pub stencil_store_op: StoreOp,
    /// Stencil clear value if `stencil_load_op` is `LoadOp::Clear`.
    pub clear_stencil: Option<u32>,
}

impl DepthAttachment {
    /// Mip 0, layer 0 of `texture`, cleared to `clear_depth` (and stencil 0) and stored.
    pub fn cleared(texture: TextureHandle, clear_depth: f32) -> Self {
        Self {
            texture,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_depth: Some(clear_depth),
            stencil_load_op: LoadOp::Clear,
            stencil_store_op: StoreOp::Store,
            clear_stencil: Some(0),
        }
    }
}

/// Most colour attachments a pass can have.
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// The colour attachments of a pass, stored inline (no allocation; RHI-11).
pub type ColorAttachments = korppi_core::SmallVec<[RenderingAttachment; MAX_COLOR_ATTACHMENTS]>;

/// Dynamic rendering pass configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderingDescriptor {
    /// Render area X offset.
    pub offset_x: i32,
    /// Render area Y offset.
    pub offset_y: i32,
    /// Render area width in pixels.
    pub width: u32,
    /// Render area height in pixels.
    pub height: u32,
    /// Color render target attachments.
    pub color_attachments: ColorAttachments,
    /// Optional depth/stencil attachment.
    pub depth_attachment: Option<DepthAttachment>,
}

/// Parameters for copying buffer bytes into a texture sub-resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferTextureCopy {
    /// Starting byte offset in the source buffer.
    pub buffer_offset: u64,
    /// Stride in bytes per row of pixels (must align to 4 bytes).
    pub buffer_row_pitch: u32,
    /// Target texture mipmap level.
    pub texture_mip_level: u32,
    /// Target texture array layer.
    pub texture_layer: u32,
    /// Top-left XYZ origin in texture space.
    pub origin: [u32; 3],
    /// Width, height, and depth extent to copy.
    pub extent: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_initial_data_may_be_shorter_but_not_longer() {
        let desc = BufferDescriptor {
            size: 16,
            usage: BufferUsage::VERTEX,
            memory: MemoryUsage::CpuToGpu,
            label: None,
        };
        assert!(desc.validate_initial_data(&[0; 16]).is_ok());
        assert!(desc.validate_initial_data(&[0; 4]).is_ok());
        assert!(desc.validate_initial_data(&[0; 17]).is_err());
    }

    #[test]
    fn texture_initial_data_must_be_exactly_one_level() {
        let desc = TextureDescriptor {
            width: 4,
            height: 2,
            depth: 1,
            array_layers: 1,
            mip_levels: 3,
            sample_count: SampleCount::Count1,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsage::SAMPLED,
            dimension: TextureDimension::D2,
            label: None,
        };
        assert_eq!(desc.level0_size_bytes(), 64);
        assert!(desc.validate_initial_data(&[0; 64]).is_ok());
        assert!(desc.validate_initial_data(&[0; 63]).is_err());
        assert!(desc.validate_initial_data(&[0; 65]).is_err());
    }
}
