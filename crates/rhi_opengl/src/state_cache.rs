//! OpenGL state cache: skips driver calls that would not change context state (RGL-15).
//!
//! It covers every piece of context state the backend sets while playing back commands.
//! Each entry is `None` while unknown:
//! - when the cache is created;
//! - after [`GlStateCache::invalidate`], which code that changes GL state behind the cache calls
//!   (`OpenGlDevice::with_gl`);
//! - for objects being deleted ([`GlStateCache::forget_buffer`] and its siblings): GL unbinds a
//!   deleted object and may give its name to a new one.
//!
//! An unknown entry is always applied, so the cache never skips a call it cannot prove
//! redundant.

use crate::gl_fns::GlExtraFns;
use crate::resource::MAX_VERTEX_BINDINGS;
use glow::HasContext;

/// Binding points per indexed target: textures, samplers, image units and uniform and storage
/// buffers. The RHI numbers binding points with a `u8` (`GlBindingTable`), so every point it
/// uses is cached.
pub const MAX_BINDING_POINTS: usize = 256;

/// Draw buffers whose blend state and write masks are cached (the RHI's attachment limit).
pub const MAX_DRAW_BUFFERS: usize = korppi_rhi::MAX_COLOR_ATTACHMENTS;

/// Blend factors and equations of one draw buffer (`glBlendFuncSeparatei`,
/// `glBlendEquationSeparatei`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlBlend {
    /// Source factor for colour.
    pub src_color: u32,
    /// Destination factor for colour.
    pub dst_color: u32,
    /// Source factor for alpha.
    pub src_alpha: u32,
    /// Destination factor for alpha.
    pub dst_alpha: u32,
    /// Colour equation.
    pub color_op: u32,
    /// Alpha equation.
    pub alpha_op: u32,
}

/// Stencil test of one face (`glStencilFuncSeparate`, `glStencilOpSeparate`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlStencilFace {
    /// Comparison function.
    pub func: u32,
    /// Reference value.
    pub reference: i32,
    /// Compare (read) mask.
    pub read_mask: u32,
    /// Operation when the stencil test fails.
    pub fail: u32,
    /// Operation when the stencil test passes and the depth test fails.
    pub depth_fail: u32,
    /// Operation when both tests pass.
    pub pass: u32,
}

/// Depth bias (`glPolygonOffset`, or `glPolygonOffsetClamp` with a non-zero clamp).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GlDepthBias {
    /// Slope-scaled factor.
    pub slope: f32,
    /// Constant units.
    pub constant: f32,
    /// Maximum bias; 0 for none.
    pub clamp: f32,
}

/// A texture level on an image unit (`glBindImageTexture`, level 0, layer 0).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlImageBinding {
    /// The texture.
    pub texture: glow::Texture,
    /// Whether every layer is bound.
    pub layered: bool,
    /// `READ_ONLY`, `WRITE_ONLY` or `READ_WRITE`.
    pub access: u32,
    /// Internal format the shader declares.
    pub format: u32,
}

/// A buffer range on an indexed binding point: (buffer, offset, size).
type BufferRange = (glow::Buffer, i32, i32);

/// Face index of front-face stencil state.
pub const FRONT: usize = 0;
/// Face index of back-face stencil state.
pub const BACK: usize = 1;
const FACES: [u32; 2] = [glow::FRONT, glow::BACK];

/// Stores `value` in `slot` and returns whether it differs from what was known.
#[inline]
fn changed<T: PartialEq>(slot: &mut Option<T>, value: T) -> bool {
    if slot.as_ref() == Some(&value) {
        false
    } else {
        *slot = Some(value);
        true
    }
}

/// Sets `cap` on or off.
///
/// # Safety
/// The context must be current on this thread.
unsafe fn set_capability(gl: &glow::Context, cap: u32, enabled: bool) {
    // SAFETY: forwarded from the caller.
    unsafe {
        if enabled {
            gl.enable(cap);
        } else {
            gl.disable(cap);
        }
    }
}

/// Unknown entries for every binding point of one indexed target.
fn unknown_points<T: Copy>() -> [Option<T>; MAX_BINDING_POINTS] {
    [None; MAX_BINDING_POINTS]
}

/// Caches OpenGL 4.5 context state to prune redundant driver calls.
pub struct GlStateCache {
    program: Option<Option<glow::Program>>,
    vao: Option<Option<glow::VertexArray>>,
    fbo: Option<Option<glow::Framebuffer>>,
    viewport: Option<[i32; 4]>,
    depth_range: Option<[f32; 2]>,
    scissor_test: Option<bool>,
    scissor: Option<[i32; 4]>,
    depth_test: Option<bool>,
    depth_write: Option<bool>,
    depth_func: Option<u32>,
    cull_face: Option<bool>,
    cull_mode: Option<u32>,
    front_face: Option<u32>,
    polygon_mode: Option<u32>,
    polygon_offset_enabled: Option<bool>,
    polygon_offset: Option<GlDepthBias>,
    depth_clamp: Option<bool>,
    alpha_to_coverage: Option<bool>,
    blend: [Option<Option<GlBlend>>; MAX_DRAW_BUFFERS],
    color_mask: [Option<[bool; 4]>; MAX_DRAW_BUFFERS],
    blend_color: Option<[f32; 4]>,
    stencil_test: Option<bool>,
    stencil: [Option<GlStencilFace>; 2],
    stencil_write_mask: [Option<u32>; 2],
    textures: [Option<Option<glow::Texture>>; MAX_BINDING_POINTS],
    samplers: [Option<Option<glow::Sampler>>; MAX_BINDING_POINTS],
    images: [Option<Option<GlImageBinding>>; MAX_BINDING_POINTS],
    uniform_buffers: [Option<Option<BufferRange>>; MAX_BINDING_POINTS],
    storage_buffers: [Option<Option<BufferRange>>; MAX_BINDING_POINTS],
    /// The VAO whose attachments `vertex_buffers` and `element_buffer` describe; attachments
    /// are VAO state, so they are unknown for any other VAO.
    attachments_vao: Option<glow::VertexArray>,
    vertex_buffers: [Option<(glow::Buffer, i32, i32)>; MAX_VERTEX_BINDINGS],
    element_buffer: Option<glow::Buffer>,
}

impl Default for GlStateCache {
    fn default() -> Self {
        Self {
            program: None,
            vao: None,
            fbo: None,
            viewport: None,
            depth_range: None,
            scissor_test: None,
            scissor: None,
            depth_test: None,
            depth_write: None,
            depth_func: None,
            cull_face: None,
            cull_mode: None,
            front_face: None,
            polygon_mode: None,
            polygon_offset_enabled: None,
            polygon_offset: None,
            depth_clamp: None,
            alpha_to_coverage: None,
            blend: [None; MAX_DRAW_BUFFERS],
            color_mask: [None; MAX_DRAW_BUFFERS],
            blend_color: None,
            stencil_test: None,
            stencil: [None; 2],
            stencil_write_mask: [None; 2],
            textures: unknown_points(),
            samplers: unknown_points(),
            images: unknown_points(),
            uniform_buffers: unknown_points(),
            storage_buffers: unknown_points(),
            attachments_vao: None,
            vertex_buffers: [None; MAX_VERTEX_BINDINGS],
            element_buffer: None,
        }
    }
}

// Every `unsafe fn` below requires the OpenGL context to be current on this thread and the
// cache to be that context's.
impl GlStateCache {
    /// Creates a cache that knows nothing about the context's state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forgets all state: the next call of every setter reaches GL. For code that changed
    /// context state without going through the cache.
    pub fn invalidate(&mut self) {
        *self = Self::default();
    }

    /// The bound VAO, if known.
    pub fn vao(&self) -> Option<Option<glow::VertexArray>> {
        self.vao
    }

    /// Uses `program` (`glUseProgram`).
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_program(&mut self, gl: &glow::Context, program: Option<glow::Program>) {
        if changed(&mut self.program, program) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.use_program(program) };
        }
    }

    /// Binds a vertex array object.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_vao(&mut self, gl: &glow::Context, vao: Option<glow::VertexArray>) {
        if changed(&mut self.vao, vao) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.bind_vertex_array(vao) };
        }
    }

    /// Binds a framebuffer for drawing and reading.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_fbo(&mut self, gl: &glow::Context, fbo: Option<glow::Framebuffer>) {
        if changed(&mut self.fbo, fbo) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, fbo) };
        }
    }

    /// Sets the viewport rectangle.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_viewport(
        &mut self,
        gl: &glow::Context,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    ) {
        if changed(&mut self.viewport, [x, y, width, height]) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.viewport(x, y, width, height) };
        }
    }

    /// Sets the depth range NDC depth maps to (`glDepthRangef`); `near` may exceed `far`.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_depth_range(&mut self, gl: &glow::Context, near: f32, far: f32) {
        if changed(&mut self.depth_range, [near, far]) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.depth_range_f32(near, far) };
        }
    }

    /// Enables the scissor test with the given rectangle.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_scissor(
        &mut self,
        gl: &glow::Context,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe {
            if changed(&mut self.scissor_test, true) {
                gl.enable(glow::SCISSOR_TEST);
            }
            if changed(&mut self.scissor, [x, y, width, height]) {
                gl.scissor(x, y, width, height);
            }
        }
    }

    /// Disables the scissor test (clears and blits must cover the whole area, RGL-09).
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn disable_scissor(&mut self, gl: &glow::Context) {
        if changed(&mut self.scissor_test, false) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.disable(glow::SCISSOR_TEST) };
        }
    }

    /// Sets the depth write mask.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_depth_write(&mut self, gl: &glow::Context, write: bool) {
        if changed(&mut self.depth_write, write) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.depth_mask(write) };
        }
    }

    /// Sets the depth test, write mask and comparison.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_depth_test(
        &mut self,
        gl: &glow::Context,
        enabled: bool,
        write: bool,
        func: u32,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe {
            if changed(&mut self.depth_test, enabled) {
                set_capability(gl, glow::DEPTH_TEST, enabled);
            }
            self.set_depth_write(gl, write);
            if changed(&mut self.depth_func, func) {
                gl.depth_func(func);
            }
        }
    }

    /// Sets face culling (`None`: off) and the front-face winding.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_cull_face(
        &mut self,
        gl: &glow::Context,
        cull_mode: Option<u32>,
        front_face: u32,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe {
            if changed(&mut self.cull_face, cull_mode.is_some()) {
                set_capability(gl, glow::CULL_FACE, cull_mode.is_some());
            }
            if let Some(mode) = cull_mode {
                if changed(&mut self.cull_mode, mode) {
                    gl.cull_face(mode);
                }
            }
            if changed(&mut self.front_face, front_face) {
                gl.front_face(front_face);
            }
        }
    }

    /// Sets the polygon rasterization mode of both faces.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_polygon_mode(&mut self, gl: &glow::Context, mode: u32) {
        if changed(&mut self.polygon_mode, mode) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.polygon_mode(glow::FRONT_AND_BACK, mode) };
        }
    }

    /// Sets the depth bias of every polygon mode; `None` disables it. A non-zero clamp needs
    /// `glPolygonOffsetClamp` (pipelines with one are only created where it exists, RHI-04).
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_depth_bias(
        &mut self,
        gl: &glow::Context,
        extra: &GlExtraFns,
        bias: Option<GlDepthBias>,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe {
            if changed(&mut self.polygon_offset_enabled, bias.is_some()) {
                for cap in [
                    glow::POLYGON_OFFSET_FILL,
                    glow::POLYGON_OFFSET_LINE,
                    glow::POLYGON_OFFSET_POINT,
                ] {
                    set_capability(gl, cap, bias.is_some());
                }
            }
            let Some(bias) = bias else {
                return;
            };
            if changed(&mut self.polygon_offset, bias) {
                // `glPolygonOffset` is `glPolygonOffsetClamp` with a zero clamp.
                match extra.polygon_offset_clamp {
                    Some(offset_clamp) if bias.clamp != 0.0 => {
                        offset_clamp(bias.slope, bias.constant, bias.clamp)
                    }
                    _ => gl.polygon_offset(bias.slope, bias.constant),
                }
            }
        }
    }

    /// Sets depth clamping.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_depth_clamp(&mut self, gl: &glow::Context, enabled: bool) {
        if changed(&mut self.depth_clamp, enabled) {
            // SAFETY: forwarded from the caller.
            unsafe { set_capability(gl, glow::DEPTH_CLAMP, enabled) };
        }
    }

    /// Sets alpha-to-coverage.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_alpha_to_coverage(&mut self, gl: &glow::Context, enabled: bool) {
        if changed(&mut self.alpha_to_coverage, enabled) {
            // SAFETY: forwarded from the caller.
            unsafe { set_capability(gl, glow::SAMPLE_ALPHA_TO_COVERAGE, enabled) };
        }
    }

    /// Sets the blending of draw buffer `buffer` (`None`: off). Buffers past
    /// [`MAX_DRAW_BUFFERS`] are set without caching.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_blend(&mut self, gl: &glow::Context, buffer: u32, blend: Option<GlBlend>) {
        let mut unknown = None;
        let slot = self.blend.get_mut(buffer as usize).unwrap_or(&mut unknown);
        let previous = *slot;
        if !changed(slot, blend) {
            return;
        }
        // SAFETY: forwarded from the caller.
        unsafe {
            match blend {
                Some(b) => {
                    if !matches!(previous, Some(Some(_))) {
                        gl.enable_draw_buffer(glow::BLEND, buffer);
                    }
                    if previous != Some(Some(b)) {
                        gl.blend_func_separate_draw_buffer(
                            buffer,
                            b.src_color,
                            b.dst_color,
                            b.src_alpha,
                            b.dst_alpha,
                        );
                        gl.blend_equation_separate_draw_buffer(buffer, b.color_op, b.alpha_op);
                    }
                }
                None => gl.disable_draw_buffer(glow::BLEND, buffer),
            }
        }
    }

    /// Sets the colour write mask (r, g, b, a) of draw buffer `buffer`.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_color_mask(&mut self, gl: &glow::Context, buffer: u32, mask: [bool; 4]) {
        let mut unknown = None;
        let slot = self
            .color_mask
            .get_mut(buffer as usize)
            .unwrap_or(&mut unknown);
        if changed(slot, mask) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.color_mask_draw_buffer(buffer, mask[0], mask[1], mask[2], mask[3]) };
        }
    }

    /// Sets the blend constant.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_blend_color(&mut self, gl: &glow::Context, color: [f32; 4]) {
        if changed(&mut self.blend_color, color) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.blend_color(color[0], color[1], color[2], color[3]) };
        }
    }

    /// Sets the stencil test.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_stencil_test(&mut self, gl: &glow::Context, enabled: bool) {
        if changed(&mut self.stencil_test, enabled) {
            // SAFETY: forwarded from the caller.
            unsafe { set_capability(gl, glow::STENCIL_TEST, enabled) };
        }
    }

    /// Sets the stencil function and operations of `face` ([`FRONT`] or [`BACK`]).
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_stencil_face(&mut self, gl: &glow::Context, face: usize, s: GlStencilFace) {
        let previous = self.stencil[face];
        if !changed(&mut self.stencil[face], s) {
            return;
        }
        let gl_face = FACES[face];
        // SAFETY: forwarded from the caller.
        unsafe {
            let func = (s.func, s.reference, s.read_mask);
            if previous.map(|p| (p.func, p.reference, p.read_mask)) != Some(func) {
                gl.stencil_func_separate(gl_face, s.func, s.reference, s.read_mask);
            }
            let ops = (s.fail, s.depth_fail, s.pass);
            if previous.map(|p| (p.fail, p.depth_fail, p.pass)) != Some(ops) {
                gl.stencil_op_separate(gl_face, s.fail, s.depth_fail, s.pass);
            }
        }
    }

    /// Sets the stencil write mask of `face` ([`FRONT`] or [`BACK`]).
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_stencil_write_mask(&mut self, gl: &glow::Context, face: usize, mask: u32) {
        if changed(&mut self.stencil_write_mask[face], mask) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.stencil_mask_separate(FACES[face], mask) };
        }
    }

    /// Binds `texture` to texture unit `unit` (`glBindTextureUnit`). Units past
    /// [`MAX_BINDING_POINTS`] are bound without caching.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn bind_texture_unit(
        &mut self,
        gl: &glow::Context,
        unit: u32,
        texture: Option<glow::Texture>,
    ) {
        let mut unknown = None;
        let slot = self.textures.get_mut(unit as usize).unwrap_or(&mut unknown);
        if changed(slot, texture) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.bind_texture_unit(unit, texture) };
        }
    }

    /// Binds `sampler` to texture unit `unit`.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn bind_sampler(
        &mut self,
        gl: &glow::Context,
        unit: u32,
        sampler: Option<glow::Sampler>,
    ) {
        let mut unknown = None;
        let slot = self.samplers.get_mut(unit as usize).unwrap_or(&mut unknown);
        if changed(slot, sampler) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.bind_sampler(unit, sampler) };
        }
    }

    /// Binds level 0 of a texture to image unit `unit` (`None`: unbinds the unit).
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn bind_image(
        &mut self,
        gl: &glow::Context,
        unit: u32,
        image: Option<GlImageBinding>,
    ) {
        let mut unknown = None;
        let slot = self.images.get_mut(unit as usize).unwrap_or(&mut unknown);
        if changed(slot, image) {
            // SAFETY: forwarded from the caller.
            unsafe {
                match image {
                    Some(i) => gl.bind_image_texture(
                        unit,
                        Some(i.texture),
                        0,
                        i.layered,
                        0,
                        i.access,
                        i.format,
                    ),
                    None => {
                        gl.bind_image_texture(unit, None, 0, false, 0, glow::READ_ONLY, glow::R8)
                    }
                }
            }
        }
    }

    /// Binds a range of `buffer` to binding point `point` of `target` (`UNIFORM_BUFFER` or
    /// `SHADER_STORAGE_BUFFER`).
    ///
    /// # Safety
    /// See the `impl` block; `target` must be one of the two indexed buffer targets above.
    pub unsafe fn bind_buffer_range(
        &mut self,
        gl: &glow::Context,
        target: u32,
        point: u32,
        buffer: glow::Buffer,
        offset: i32,
        size: i32,
    ) {
        let points = if target == glow::UNIFORM_BUFFER {
            &mut self.uniform_buffers
        } else {
            &mut self.storage_buffers
        };
        let mut unknown = None;
        let slot = points.get_mut(point as usize).unwrap_or(&mut unknown);
        if changed(slot, Some((buffer, offset, size))) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.bind_buffer_range(target, point, Some(buffer), offset, size) };
        }
    }

    /// Attaches `buffer` to vertex buffer binding `binding` of `vao`.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_vertex_buffer(
        &mut self,
        gl: &glow::Context,
        vao: glow::VertexArray,
        binding: u32,
        buffer: glow::Buffer,
        offset: i32,
        stride: i32,
    ) {
        self.attachments_of(vao);
        let mut unknown = None;
        let slot = self
            .vertex_buffers
            .get_mut(binding as usize)
            .unwrap_or(&mut unknown);
        if changed(slot, (buffer, offset, stride)) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.vertex_array_vertex_buffer(vao, binding, Some(buffer), offset, stride) };
        }
    }

    /// Attaches `buffer` as the element (index) buffer of `vao`.
    ///
    /// # Safety
    /// See the `impl` block.
    pub unsafe fn set_element_buffer(
        &mut self,
        gl: &glow::Context,
        vao: glow::VertexArray,
        buffer: glow::Buffer,
    ) {
        self.attachments_of(vao);
        if changed(&mut self.element_buffer, buffer) {
            // SAFETY: forwarded from the caller.
            unsafe { gl.vertex_array_element_buffer(vao, Some(buffer)) };
        }
    }

    /// Makes the attachment entries describe `vao`, forgetting those of another VAO.
    fn attachments_of(&mut self, vao: glow::VertexArray) {
        if self.attachments_vao != Some(vao) {
            self.attachments_vao = Some(vao);
            self.vertex_buffers = [None; MAX_VERTEX_BINDINGS];
            self.element_buffer = None;
        }
    }

    /// Forgets a texture that is being deleted: GL unbinds a deleted texture from the
    /// context's texture and image units, and a later texture may reuse its name.
    pub fn forget_texture(&mut self, texture: glow::Texture) {
        for bound in &mut self.textures {
            if *bound == Some(Some(texture)) {
                *bound = None;
            }
        }
        for bound in &mut self.images {
            if matches!(bound, Some(Some(image)) if image.texture == texture) {
                *bound = None;
            }
        }
    }

    /// Forgets a sampler that is being deleted (see [`GlStateCache::forget_texture`]).
    pub fn forget_sampler(&mut self, sampler: glow::Sampler) {
        for bound in &mut self.samplers {
            if *bound == Some(Some(sampler)) {
                *bound = None;
            }
        }
    }

    /// Forgets a buffer that is being deleted: GL unbinds it from the context's binding points
    /// and the bound VAO, while other VAOs keep the deleted object attached under a name a new
    /// buffer may reuse.
    pub fn forget_buffer(&mut self, buffer: glow::Buffer) {
        for bound in self
            .uniform_buffers
            .iter_mut()
            .chain(self.storage_buffers.iter_mut())
        {
            if matches!(bound, Some(Some((b, _, _))) if *b == buffer) {
                *bound = None;
            }
        }
        for attached in &mut self.vertex_buffers {
            if matches!(attached, Some((b, _, _)) if *b == buffer) {
                *attached = None;
            }
        }
        if self.element_buffer == Some(buffer) {
            self.element_buffer = None;
        }
    }

    /// Forgets a program that is being deleted (see [`GlStateCache::forget_texture`]).
    pub fn forget_program(&mut self, program: glow::Program) {
        if self.program == Some(Some(program)) {
            self.program = None;
        }
    }

    /// Forgets a vertex array object that is being deleted: GL binds VAO 0 in its place.
    pub fn forget_vao(&mut self, vao: glow::VertexArray) {
        if self.vao == Some(Some(vao)) {
            self.vao = None;
        }
        if self.attachments_vao == Some(vao) {
            self.attachments_vao = None;
            self.vertex_buffers = [None; MAX_VERTEX_BINDINGS];
            self.element_buffer = None;
        }
    }

    /// Forgets a framebuffer that is being deleted: GL binds framebuffer 0 in its place.
    pub fn forget_fbo(&mut self, fbo: glow::Framebuffer) {
        if self.fbo == Some(Some(fbo)) {
            self.fbo = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GlContext;
    use std::fmt::Debug;

    /// Checks one cached state: `apply` sets it to `expected`; after `tamper` changes it behind
    /// the cache, applying the same value again is skipped (GL still reports `tampered`), and
    /// after `invalidate` it is applied again.
    fn check<T: PartialEq + Debug>(
        what: &str,
        cache: &mut GlStateCache,
        apply: impl Fn(&mut GlStateCache),
        tamper: impl Fn(),
        observe: impl Fn() -> T,
        expected: T,
        tampered: T,
    ) {
        apply(cache);
        assert_eq!(observe(), expected, "{what}: applied");
        tamper();
        assert_eq!(observe(), tampered, "{what}: changed behind the cache");
        apply(cache);
        assert_eq!(observe(), tampered, "{what}: a redundant call is skipped");
        cache.invalidate();
        apply(cache);
        assert_eq!(
            observe(),
            expected,
            "{what}: applied again after invalidate"
        );
    }

    #[test]
    fn every_cached_state_skips_redundant_calls_and_honours_invalidate() {
        let context = GlContext::create_headless().expect("headless context");
        let gl = &context.gl;
        let extra = &context.extra;
        let mut cache = GlStateCache::new();
        let c = &mut cache;
        // SAFETY: the context is current on this thread for the whole test; every object used
        // is created here.
        unsafe {
            let int = |p| gl.get_parameter_i32(p);
            let ints4 = |p| {
                let mut v = [0; 4];
                gl.get_parameter_i32_slice(p, &mut v);
                v
            };
            let floats = |p| {
                let mut v = [0.0_f32; 4];
                gl.get_parameter_f32_slice(p, &mut v);
                v
            };
            let indexed = |p, i| gl.get_parameter_indexed_i32(p, i);
            let name = |n: std::num::NonZeroU32| n.get() as i32;

            check(
                "viewport",
                c,
                |c| c.set_viewport(gl, 1, 2, 30, 40),
                || gl.viewport(0, 0, 1, 1),
                || ints4(glow::VIEWPORT),
                [1, 2, 30, 40],
                [0, 0, 1, 1],
            );
            check(
                "depth range",
                c,
                |c| c.set_depth_range(gl, 0.25, 0.75),
                || gl.depth_range_f32(0.0, 1.0),
                || {
                    let mut v = [0.0_f32; 2];
                    gl.get_parameter_f32_slice(glow::DEPTH_RANGE, &mut v);
                    v
                },
                [0.25, 0.75],
                [0.0, 1.0],
            );
            check(
                "scissor",
                c,
                |c| c.set_scissor(gl, 3, 4, 5, 6),
                || gl.disable(glow::SCISSOR_TEST),
                || gl.is_enabled(glow::SCISSOR_TEST),
                true,
                false,
            );
            let blend = GlBlend {
                src_color: glow::SRC_ALPHA,
                dst_color: glow::ONE_MINUS_SRC_ALPHA,
                src_alpha: glow::ONE,
                dst_alpha: glow::ZERO,
                color_op: glow::FUNC_ADD,
                alpha_op: glow::MAX,
            };
            check(
                "blend enable",
                c,
                |c| c.set_blend(gl, 0, Some(blend)),
                || gl.disable_draw_buffer(glow::BLEND, 0),
                || gl.is_enabled(glow::BLEND),
                true,
                false,
            );
            check(
                "blend factors",
                c,
                |c| c.set_blend(gl, 0, Some(blend)),
                || {
                    gl.blend_func_separate_draw_buffer(
                        0,
                        glow::ONE,
                        glow::ONE,
                        glow::ONE,
                        glow::ONE,
                    )
                },
                || [int(glow::BLEND_SRC_RGB), int(glow::BLEND_DST_RGB)],
                [glow::SRC_ALPHA as i32, glow::ONE_MINUS_SRC_ALPHA as i32],
                [glow::ONE as i32, glow::ONE as i32],
            );
            check(
                "colour mask",
                c,
                |c| c.set_color_mask(gl, 0, [true, false, true, false]),
                || gl.color_mask_draw_buffer(0, false, false, false, false),
                || ints4(glow::COLOR_WRITEMASK),
                [1, 0, 1, 0],
                [0, 0, 0, 0],
            );
            check(
                "blend colour",
                c,
                |c| c.set_blend_color(gl, [0.5, 0.25, 0.0, 1.0]),
                || gl.blend_color(0.0, 0.0, 0.0, 0.0),
                || floats(glow::BLEND_COLOR),
                [0.5, 0.25, 0.0, 1.0],
                [0.0; 4],
            );
            let stencil = GlStencilFace {
                func: glow::EQUAL,
                reference: 7,
                read_mask: 0x0F,
                fail: glow::KEEP,
                depth_fail: glow::INCR,
                pass: glow::REPLACE,
            };
            check(
                "stencil function",
                c,
                |c| c.set_stencil_face(gl, BACK, stencil),
                || gl.stencil_func_separate(glow::BACK, glow::ALWAYS, 0, 0xFF),
                // The reference reads back clamped to the framebuffer's stencil bits (none here).
                || {
                    [
                        int(glow::STENCIL_BACK_FUNC),
                        int(glow::STENCIL_BACK_VALUE_MASK),
                    ]
                },
                [glow::EQUAL as i32, 0x0F],
                [glow::ALWAYS as i32, 0xFF],
            );
            check(
                "stencil operations",
                c,
                |c| c.set_stencil_face(gl, FRONT, stencil),
                || gl.stencil_op_separate(glow::FRONT, glow::KEEP, glow::KEEP, glow::KEEP),
                || int(glow::STENCIL_PASS_DEPTH_PASS),
                glow::REPLACE as i32,
                glow::KEEP as i32,
            );
            check(
                "stencil write mask",
                c,
                |c| c.set_stencil_write_mask(gl, FRONT, 0x3C),
                || gl.stencil_mask_separate(glow::FRONT, 0xFF),
                || int(glow::STENCIL_WRITEMASK),
                0x3C,
                0xFF,
            );
            check(
                "polygon mode",
                c,
                |c| c.set_polygon_mode(gl, glow::LINE),
                || gl.polygon_mode(glow::FRONT_AND_BACK, glow::FILL),
                // Mesa writes two values (front and back).
                || ints4(glow::POLYGON_MODE)[0],
                glow::LINE as i32,
                glow::FILL as i32,
            );
            let bias = GlDepthBias {
                slope: 2.0,
                constant: 3.0,
                clamp: 0.0,
            };
            check(
                "depth bias",
                c,
                |c| c.set_depth_bias(gl, extra, Some(bias)),
                || gl.polygon_offset(0.0, 0.0),
                || floats(glow::POLYGON_OFFSET_FACTOR)[0],
                2.0,
                0.0,
            );
            check(
                "depth clamp",
                c,
                |c| c.set_depth_clamp(gl, true),
                || gl.disable(glow::DEPTH_CLAMP),
                || gl.is_enabled(glow::DEPTH_CLAMP),
                true,
                false,
            );

            // Indexed bindings, including units the old cache dropped (>= 32).
            let buffer = gl.create_named_buffer().expect("buffer");
            gl.named_buffer_storage(buffer, 1024, None, 0);
            for (target, query) in [
                (glow::UNIFORM_BUFFER, glow::UNIFORM_BUFFER_BINDING),
                (
                    glow::SHADER_STORAGE_BUFFER,
                    glow::SHADER_STORAGE_BUFFER_BINDING,
                ),
            ] {
                check(
                    "buffer range",
                    c,
                    |c| c.bind_buffer_range(gl, target, 5, buffer, 256, 512),
                    || gl.bind_buffer_base(target, 5, None),
                    || indexed(query, 5),
                    name(buffer.0),
                    0,
                );
            }
            let texture = gl.create_named_texture(glow::TEXTURE_2D).expect("texture");
            gl.texture_storage_2d(texture, 1, glow::RGBA8, 4, 4);
            let unit_binding = |unit: u32| {
                gl.active_texture(glow::TEXTURE0 + unit);
                let bound = (int(glow::TEXTURE_BINDING_2D), int(glow::SAMPLER_BINDING));
                gl.active_texture(glow::TEXTURE0);
                bound
            };
            let sampler = gl.create_sampler().expect("sampler");
            for unit in [0, 31, 32, 40] {
                check(
                    "texture unit",
                    c,
                    |c| {
                        c.bind_texture_unit(gl, unit, Some(texture));
                        c.bind_sampler(gl, unit, Some(sampler));
                    },
                    || {
                        gl.bind_texture_unit(unit, None);
                        gl.bind_sampler(unit, None);
                    },
                    || unit_binding(unit),
                    (name(texture.0), name(sampler.0)),
                    (0, 0),
                );
            }
            let image = GlImageBinding {
                texture,
                layered: false,
                access: glow::WRITE_ONLY,
                format: glow::RGBA8,
            };
            check(
                "image unit",
                c,
                |c| c.bind_image(gl, 3, Some(image)),
                || gl.bind_image_texture(3, None, 0, false, 0, glow::READ_ONLY, glow::R8),
                || indexed(glow::IMAGE_BINDING_NAME, 3),
                name(texture.0),
                0,
            );

            // VAO attachments are per VAO.
            let vao = gl.create_named_vertex_array().expect("vao");
            check(
                "element buffer",
                c,
                |c| c.set_element_buffer(gl, vao, buffer),
                || gl.vertex_array_element_buffer(vao, None),
                || {
                    gl.bind_vertex_array(Some(vao));
                    int(glow::ELEMENT_ARRAY_BUFFER_BINDING)
                },
                name(buffer.0),
                0,
            );
            gl.delete_vertex_array(vao);
            gl.delete_sampler(sampler);
            gl.delete_texture(texture);
            gl.delete_buffer(buffer);
        }
    }

    #[test]
    fn forgotten_objects_are_bound_again() {
        let context = GlContext::create_headless().expect("headless context");
        let gl = &context.gl;
        let mut cache = GlStateCache::new();
        // SAFETY: the context is current on this thread; every object used is created here.
        unsafe {
            let buffer = gl.create_named_buffer().expect("buffer");
            gl.named_buffer_storage(buffer, 256, None, 0);
            let bound = || gl.get_parameter_indexed_i32(glow::UNIFORM_BUFFER_BINDING, 2);
            cache.bind_buffer_range(gl, glow::UNIFORM_BUFFER, 2, buffer, 0, 256);
            // As if the buffer were deleted (GL unbinds it) and its name reused.
            gl.bind_buffer_base(glow::UNIFORM_BUFFER, 2, None);
            cache.forget_buffer(buffer);
            cache.bind_buffer_range(gl, glow::UNIFORM_BUFFER, 2, buffer, 0, 256);
            assert_eq!(bound(), buffer.0.get() as i32);

            let texture = gl.create_named_texture(glow::TEXTURE_2D).expect("texture");
            gl.texture_storage_2d(texture, 1, glow::RGBA8, 4, 4);
            let image = GlImageBinding {
                texture,
                layered: false,
                access: glow::READ_ONLY,
                format: glow::RGBA8,
            };
            cache.bind_image(gl, 1, Some(image));
            gl.bind_image_texture(1, None, 0, false, 0, glow::READ_ONLY, glow::R8);
            cache.forget_texture(texture);
            cache.bind_image(gl, 1, Some(image));
            assert_eq!(
                gl.get_parameter_indexed_i32(glow::IMAGE_BINDING_NAME, 1),
                texture.0.get() as i32
            );

            let vao = gl.create_named_vertex_array().expect("vao");
            cache.set_vao(gl, Some(vao));
            gl.bind_vertex_array(None);
            cache.forget_vao(vao);
            cache.set_vao(gl, Some(vao));
            assert_eq!(
                gl.get_parameter_i32(glow::VERTEX_ARRAY_BINDING),
                vao.0.get() as i32
            );
            gl.delete_vertex_array(vao);
            gl.delete_texture(texture);
            gl.delete_buffer(buffer);
        }
    }
}
