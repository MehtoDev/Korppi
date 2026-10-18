//! Presentation to OS windows (finding RHI-02).
//!
//! Swapchains are device-owned objects referenced by a [`SwapchainHandle`], like every other
//! resource. A frame presents like this:
//!
//! 1. [`GraphicsDevice::acquire_next_image`](crate::GraphicsDevice::acquire_next_image) returns a
//!    [`SwapchainImage`], or `None` while the window has no area (minimized): skip the frame.
//! 2. Record commands that render into [`SwapchainImage::texture`] and submit them. The image's
//!    contents are undefined after acquisition, so transition it from `ImageLayout::Undefined`
//!    like a new texture. The backend makes the first submission that uses the image wait until
//!    the presentation engine has released it.
//! 3. [`GraphicsDevice::present`](crate::GraphicsDevice::present) queues the image for display
//!    after every earlier submission. The image may be left in any layout: the backend moves it
//!    to the presentation layout if the commands did not.
//!
//! Window resizes are reported with
//! [`GraphicsDevice::resize_swapchain`](crate::GraphicsDevice::resize_swapchain). Backends also
//! recreate a swapchain the window system reports as out of date or suboptimal on their own, at
//! the next acquisition. Swapchain textures belong to the swapchain: they cannot be destroyed and
//! become invalid when it is resized, recreated or destroyed.

use crate::handle::{SwapchainHandle, TextureHandle};
use crate::types::{TextureFormat, TextureUsage};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// A connection to a window system, which presentation-capable OpenGL devices are created for.
///
/// Implemented for every `HasDisplayHandle + Send + Sync` type, such as a `winit` window or
/// `winit::event_loop::OwnedDisplayHandle`. Devices keep the target alive (through an `Arc`) for
/// as long as they use its display connection.
pub trait DisplayTarget: HasDisplayHandle + Send + Sync {}

impl<T: HasDisplayHandle + Send + Sync + ?Sized> DisplayTarget for T {}

/// A window a swapchain can present to.
///
/// Implemented for every `HasWindowHandle + HasDisplayHandle + Send + Sync` type, such as a
/// `winit` window. Swapchains keep the window alive (through an `Arc`) until they are destroyed.
/// An `Arc<dyn PresentTarget>` also coerces to the `Arc<dyn DisplayTarget>` a presentation-capable
/// OpenGL device is created from.
pub trait PresentTarget: DisplayTarget + HasWindowHandle {}

impl<T: HasWindowHandle + HasDisplayHandle + Send + Sync + ?Sized> PresentTarget for T {}

/// How presented images are synchronized with the display's refresh.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PresentMode {
    /// Vertical sync: images are shown in order, one per refresh. Always supported.
    #[default]
    Fifo,
    /// Vertical sync, but a late image is shown immediately (may tear). Falls back to `Fifo`.
    FifoRelaxed,
    /// The newest image replaces a waiting one at the next refresh: low latency without tearing.
    /// Falls back to `Fifo`.
    Mailbox,
    /// Images are shown immediately (may tear). Falls back to `Fifo`.
    Immediate,
}

/// Parameters of a new swapchain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwapchainDescriptor {
    /// Width in physical pixels. Ignored where the window system dictates the size.
    pub width: u32,
    /// Height in physical pixels. Ignored where the window system dictates the size.
    pub height: u32,
    /// Requested presentation mode; see [`SwapchainInfo::present_mode`] for the one in use.
    pub present_mode: PresentMode,
    /// Prefer an sRGB format, so shaders write linear colour and the hardware encodes it.
    pub srgb: bool,
    /// Minimum number of images; 0 lets the backend choose (usually 3).
    pub image_count: u32,
}

impl SwapchainDescriptor {
    /// A vsync'd sRGB swapchain of `width` x `height` pixels.
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            present_mode: PresentMode::Fifo,
            srgb: true,
            image_count: 0,
        }
    }
}

/// Properties a swapchain was created (or last recreated) with.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SwapchainInfo {
    /// Format of the swapchain textures; create pipelines that render to them with it.
    pub format: TextureFormat,
    /// Usage the swapchain textures support (always includes `COLOR_ATTACHMENT`).
    pub usage: TextureUsage,
    /// Width in pixels (0 while the window has no area).
    pub width: u32,
    /// Height in pixels (0 while the window has no area).
    pub height: u32,
    /// Presentation mode in use (the requested one, or `Fifo` if it is unsupported).
    pub present_mode: PresentMode,
    /// Number of images.
    pub image_count: u32,
}

/// An image acquired from a swapchain, to render into and then present.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SwapchainImage {
    /// Swapchain the image belongs to.
    pub swapchain: SwapchainHandle,
    /// Index of the image in the swapchain.
    pub index: u32,
    /// The image as a texture, valid until the swapchain is resized, recreated or destroyed.
    pub texture: TextureHandle,
    /// Width of the texture in pixels.
    pub width: u32,
    /// Height of the texture in pixels.
    pub height: u32,
}
