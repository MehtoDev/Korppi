//! OpenGL 4.5 Core Direct State Access (DSA) Render Hardware Interface implementation.

#![warn(missing_docs)]

/// Bind group layouts, bind groups and per-pipeline binding points.
pub mod binding;
/// Adapter information, limits, features and format support.
pub mod capabilities;
/// OpenGL command buffer packet stream and recorder.
pub mod command;
/// EGL contexts (headless or on a window system), window surfaces and the OpenGL loader.
pub mod context;
/// Concrete OpenGlDevice implementing GraphicsDevice.
pub mod device;
/// EGL contexts and window surfaces.
#[cfg(not(windows))]
mod egl;
/// Zero-churn generational LRU Framebuffer Object cache.
pub mod fbo_cache;
/// OpenGL entry points `glow` does not wrap.
pub mod gl_fns;
/// Program binaries of linked pipelines.
mod program_cache;
/// Dedicated thread that owns the OpenGL context and runs every GL call.
mod render_thread;
/// Internal OpenGL resources and handle registry.
pub mod resource;
/// SPIR-V input: translation to GLSL 4.50 with naga.
pub mod spirv;
/// Zero-allocation OpenGL state cache.
pub mod state_cache;
/// Presentation: window surfaces and swapchain textures.
mod swapchain;
/// Textures and texture views.
mod texture;
/// Recorded copies, blits and mip generation.
mod transfer;
/// Mapping from abstract RHI types to OpenGL 4.5 Core constants.
pub mod types;
/// WGL contexts and window surfaces.
#[cfg(windows)]
mod wgl;

pub use device::{OpenGlDevice, PUSH_CONSTANT_BINDING};
pub use fbo_cache::GlFboCache;
/// The OpenGL bindings `OpenGlDevice::with_gl` hands out.
pub use glow;
pub use swapchain::SWAPCHAIN_IMAGE_COUNT;
