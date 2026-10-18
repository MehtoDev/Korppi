//! Cross-backend conformance harness for `korppi_rhi` backends.
//!
//! The integration tests in this crate drive the Vulkan and OpenGL backends through the public
//! [`GraphicsDevice`] / `CommandBufferRecorder` API only and compare what they produce against the
//! RHI contract. Most tests are ported from the RHI audit probes and are named after their probe
//! and finding IDs (see `docs/audits/2026-10-10-rhi-followup-review.md`).
//!
//! # RHI contract assumed by the tests
//! - Row 0 of every texture returned by `read_texture` is the **top** of the image.
//! - Viewport and scissor rectangles have a **top-left** origin.
//! - Counter-clockwise triangles in +Y-up NDC are front-facing.
//! - Depth range is \[0, 1\].
//!
//! Every [`TestDevice`] also fails its test if the backend's API-level validation (Vulkan
//! validation layers, OpenGL debug output) reports an error; see
//! [`TestDevice::assert_no_api_errors`].
//!
//! # Presentation tests
//! `tests/present.rs` presents into X11 windows ([`display`]): on `$DISPLAY`, or on an `Xvfb`
//! server the tests start themselves, and reads back what they show. `tests/present_wayland.rs`
//! presents into Wayland windows: on `$WAYLAND_DISPLAY`, or on a headless Weston. Without a
//! display server the tests are skipped.
//!
//! # Requirements
//! `glslc` must be on `PATH` (Vulkan shaders are compiled from GLSL at test time), a Vulkan 1.3
//! driver must be installed (Mesa lavapipe is enough), and EGL must provide desktop OpenGL 4.5.

#[cfg(target_os = "linux")]
pub mod display;

use korppi_rhi::{
    AccessFlags, DiagnosticSeverity, DisplayTarget, GraphicsDevice, ImageLayout,
    ImageMemoryBarrier, LoadOp, PipelineBarrier, PipelineStageFlags, RenderingAttachment,
    RenderingDescriptor, SampleCount, ShaderDescriptor, ShaderHandle, ShaderSource, ShaderStage,
    StoreOp, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle, TextureUsage,
};
use korppi_rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupHandle, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutHandle, BindingResource, BindingType, BufferHandle,
    SamplerHandle, ShaderStages,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;
use std::io::Write;
use std::ops::Deref;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// Graphics backend under test.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// `korppi_rhi_vulkan` (Vulkan 1.3).
    Vulkan,
    /// `korppi_rhi_opengl` (OpenGL 4.5 Core, EGL).
    OpenGl,
}

impl Backend {
    /// Every backend, in a fixed order.
    pub const ALL: [Backend; 2] = [Backend::Vulkan, Backend::OpenGl];
}

/// A device opened for a test.
pub struct TestDevice {
    device: BackendDevice,
    backend: Backend,
}

/// The device of either backend.
enum BackendDevice {
    Vulkan(VulkanDevice),
    OpenGl(OpenGlDevice),
}

impl TestDevice {
    /// Opens a headless device for `backend`.
    ///
    /// # Panics
    /// Panics if the backend cannot be initialized on this machine.
    pub fn open(backend: Backend) -> Self {
        match backend {
            Backend::Vulkan => Self {
                device: BackendDevice::Vulkan(
                    VulkanDevice::create_headless().expect("headless Vulkan device creation"),
                ),
                backend,
            },
            Backend::OpenGl => Self {
                device: BackendDevice::OpenGl(
                    OpenGlDevice::create_headless().expect("headless OpenGL device creation"),
                ),
                backend,
            },
        }
    }

    /// Opens a device for `backend` that can present to windows of `display`.
    ///
    /// # Panics
    /// Panics if the backend cannot be initialized on this machine.
    pub fn open_for_display(backend: Backend, display: Arc<dyn DisplayTarget>) -> Self {
        match backend {
            Backend::Vulkan => Self::open(backend),
            Backend::OpenGl => Self {
                device: BackendDevice::OpenGl(
                    OpenGlDevice::create_for_display(display)
                        .expect("OpenGL device creation for a display"),
                ),
                backend,
            },
        }
    }

    /// Backend of this device.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The OpenGL device, for backend-specific checks.
    pub fn opengl(&self) -> Option<&OpenGlDevice> {
        match &self.device {
            BackendDevice::OpenGl(device) => Some(device),
            BackendDevice::Vulkan(_) => None,
        }
    }

    /// Creates a shader from Vulkan GLSL, compiled to SPIR-V with `glslc`. Both backends take
    /// the same SPIR-V (the OpenGL backend translates it; RHI-08).
    pub fn shader(&self, stage: ShaderStage, source: &str) -> ShaderHandle {
        let spirv = compile_spirv(source, stage);
        let source = ShaderSource::SpirvBytes(&spirv);
        (**self)
            .create_shader(&ShaderDescriptor {
                stage,
                entry_point: "main",
                source,
                label: None,
            })
            .expect("create_shader")
    }
}

/// One binding of [`TestDevice::bind_group`]: binding number, type, visible stages, resource.
pub type TestBinding = (u32, BindingType, ShaderStages, BindingResource);

/// A uniform buffer binding without a dynamic offset.
pub const UNIFORM: BindingType = BindingType::UniformBuffer {
    dynamic_offset: false,
};

/// A read-write storage buffer binding without a dynamic offset.
pub const STORAGE: BindingType = BindingType::StorageBuffer {
    read_only: false,
    dynamic_offset: false,
};

/// A whole buffer range as a binding resource.
pub fn buffer_range(buffer: BufferHandle, offset: u64, size: u64) -> BindingResource {
    BindingResource::Buffer {
        buffer,
        offset,
        size,
    }
}

/// A texture sampled in `ShaderReadOnly` as a binding resource.
pub fn sampled(texture: TextureHandle, sampler: SamplerHandle) -> BindingResource {
    BindingResource::SampledTexture {
        texture: texture.into(),
        sampler,
        layout: ImageLayout::ShaderReadOnly,
    }
}

impl TestDevice {
    /// Creates a bind group layout declaring `(binding, type, stages)` entries.
    ///
    /// # Panics
    /// Panics if the layout cannot be created.
    pub fn layout(&self, entries: &[(u32, BindingType, ShaderStages)]) -> BindGroupLayoutHandle {
        (**self)
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: entries
                    .iter()
                    .map(|&(binding, ty, stages)| BindGroupLayoutEntry {
                        binding,
                        ty,
                        stages,
                    })
                    .collect(),
                label: None,
            })
            .expect("create_bind_group_layout")
    }

    /// Creates a group of `layout` binding `(binding, resource)` entries.
    ///
    /// # Panics
    /// Panics if the group cannot be created.
    pub fn group(
        &self,
        layout: BindGroupLayoutHandle,
        entries: &[(u32, BindingResource)],
    ) -> BindGroupHandle {
        (**self)
            .create_bind_group(&BindGroupDescriptor {
                layout,
                entries: entries
                    .iter()
                    .map(|&(binding, resource)| BindGroupEntry { binding, resource })
                    .collect(),
                label: None,
            })
            .expect("create_bind_group")
    }

    /// Creates a bind group layout declaring `bindings` and a group filling it with their
    /// resources.
    ///
    /// # Panics
    /// Panics if either cannot be created.
    pub fn bind_group(&self, bindings: &[TestBinding]) -> (BindGroupLayoutHandle, BindGroupHandle) {
        let layout = self.layout(
            &bindings
                .iter()
                .map(|&(binding, ty, stages, _)| (binding, ty, stages))
                .collect::<Vec<_>>(),
        );
        let group = self.group(
            layout,
            &bindings
                .iter()
                .map(|&(binding, _, _, resource)| (binding, resource))
                .collect::<Vec<_>>(),
        );
        (layout, group)
    }
}

impl TestDevice {
    /// Panics if the backend's API-level validation (Vulkan validation layers, OpenGL debug
    /// output) reported any error since the last check.
    ///
    /// Runs automatically when the device is dropped, so every conformance test fails on API
    /// errors even if its pixel assertions pass.
    pub fn assert_no_api_errors(&self) {
        let errors: Vec<_> = (**self)
            .take_diagnostics()
            .into_iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "{:?} reported {} API error(s): {errors:#?}",
            self.backend,
            errors.len()
        );
    }
}

impl Drop for TestDevice {
    fn drop(&mut self) {
        // Do not panic while already unwinding: the original failure is the one to report.
        if !std::thread::panicking() {
            self.assert_no_api_errors();
        }
    }
}

impl Deref for TestDevice {
    type Target = dyn GraphicsDevice;

    fn deref(&self) -> &Self::Target {
        match &self.device {
            BackendDevice::Vulkan(device) => device,
            BackendDevice::OpenGl(device) => device,
        }
    }
}

/// Compiles Vulkan GLSL to SPIR-V bytes with `glslc`.
///
/// Compilations are cached per process and run one at a time: concurrent `glslc` processes were
/// observed to crash (SIGSEGV in about 9 % of parallel runs with shaderc 2026.4). A `glslc`
/// killed by a signal is retried.
///
/// # Panics
/// Panics if `glslc` is missing or the source does not compile.
pub fn compile_spirv(source: &str, stage: ShaderStage) -> Vec<u8> {
    static CACHE: Mutex<Vec<(ShaderStage, String, Vec<u8>)>> = Mutex::new(Vec::new());
    let mut cache = CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, _, spirv)) = cache
        .iter()
        .find(|(s, src, _)| *s == stage && src == source)
    {
        return spirv.clone();
    }
    let mut attempts = 0;
    let output = loop {
        let output = run_glslc(source, stage);
        attempts += 1;
        // `ExitStatus::code` is `None` when the process was killed by a signal (a crash, not a
        // compile error).
        if output.status.code().is_some() || attempts == 3 {
            break output;
        }
    };
    assert!(
        output.status.success(),
        "glslc failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    cache.push((stage, source.to_owned(), output.stdout.clone()));
    output.stdout
}

fn run_glslc(source: &str, stage: ShaderStage) -> std::process::Output {
    let stage_name = match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
        ShaderStage::Compute => "compute",
    };
    let mut child = Command::new("glslc")
        .arg(format!("-fshader-stage={stage_name}"))
        .args(["-", "-o", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("glslc must be on PATH to run the RHI conformance suite");
    child
        .stdin
        .as_mut()
        .expect("glslc stdin")
        .write_all(source.as_bytes())
        .expect("write shader source to glslc");
    child.wait_with_output().expect("wait for glslc")
}

/// Creates an `Rgba8Unorm` texture usable as a colour attachment and a copy source.
pub fn color_target(
    device: &dyn GraphicsDevice,
    width: u32,
    height: u32,
    extra_usage: TextureUsage,
) -> TextureHandle {
    device
        .create_texture(
            &TextureDescriptor {
                width,
                height,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC | extra_usage,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("create colour target")
}

/// Barrier that moves a freshly created texture into the colour-attachment layout.
pub fn undefined_to_color_attachment(texture: TextureHandle) -> PipelineBarrier {
    PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
        texture,
        range: korppi_rhi::TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::ColorAttachment,
    })
}

/// A single-colour-attachment pass over the whole `width` x `height` target.
pub fn color_pass(
    texture: TextureHandle,
    width: u32,
    height: u32,
    load_op: LoadOp,
    clear_color: [f32; 4],
) -> RenderingDescriptor {
    RenderingDescriptor {
        offset_x: 0,
        offset_y: 0,
        width,
        height,
        color_attachments: korppi_rhi::smallvec![RenderingAttachment {
            texture,
            mip_level: 0,
            layer: 0,
            load_op,
            store_op: StoreOp::Store,
            clear_color: Some(clear_color),
            resolve_target: None,
        }],
        depth_attachment: None,
    }
}

/// An RGBA8 image read back from a texture, rows in the order `read_texture` returned them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Tightly packed RGBA8 texels, row 0 first.
    pub data: Vec<u8>,
}

impl Image {
    /// Reads mip 0 of an `Rgba8Unorm` texture.
    ///
    /// # Panics
    /// Panics if the readback fails or returns the wrong number of bytes.
    pub fn read(
        device: &dyn GraphicsDevice,
        texture: TextureHandle,
        width: u32,
        height: u32,
    ) -> Self {
        let data = device.read_texture(texture, 0).expect("read_texture");
        assert_eq!(
            data.len(),
            (width * height * 4) as usize,
            "read_texture returned an unexpected byte count"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Texel at column `x`, row `y` (row 0 = first row returned).
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ]
    }

    /// Number of texels for which `predicate` holds.
    pub fn count(&self, predicate: impl Fn([u8; 4]) -> bool) -> usize {
        self.data
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| predicate(**p))
            .count()
    }

    /// First and last row index containing a texel for which `predicate` holds.
    pub fn rows_where(&self, predicate: impl Fn([u8; 4]) -> bool) -> Option<(u32, u32)> {
        let mut first = None;
        let mut last = None;
        for y in 0..self.height {
            if (0..self.width).any(|x| predicate(self.pixel(x, y))) {
                first.get_or_insert(y);
                last = Some(y);
            }
        }
        first.zip(last)
    }
}

/// True if every channel of `actual` is within `tolerance` of `expected`.
pub fn approx_rgba(actual: [u8; 4], expected: [u8; 4], tolerance: u8) -> bool {
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| a.abs_diff(*e) <= tolerance)
}

/// Converts a normalized colour to RGBA8 with round-to-nearest.
pub fn to_rgba8(color: [f32; 4]) -> [u8; 4] {
    color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
}
