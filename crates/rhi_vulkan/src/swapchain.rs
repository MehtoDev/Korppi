//! Window surfaces and swapchains (FIX-1.3; findings RHI-02, RVK-02).
//!
//! Follows the Vulkan specification and the Vulkan Samples / Vulkan Guide swapchain guidance:
//! - The surface is created from `raw-window-handle` handles with `ash-window`, and the device's
//!   queue family must support presenting to it (`vkGetPhysicalDeviceSurfaceSupportKHR`).
//! - Format and colour space, present mode, image count, extent, pre-transform and composite
//!   alpha are chosen from the surface queries (the `choose_*` functions), never assumed.
//! - Each acquisition signals a binary semaphore taken from a pool. The first submission that
//!   uses the image waits for it at `COLOR_ATTACHMENT_OUTPUT` (the recorder adds that stage to
//!   the source scope of barriers on swapchain images, which chains their layout transitions
//!   after the wait); if no submission used the image, the present batch waits for it. The
//!   semaphore returns to the pool once that submission has completed.
//! - `present` submits a batch that moves the image to `PRESENT_SRC_KHR` if the caller's
//!   commands left it in another layout, and signals the image's own present semaphore (one per
//!   swapchain image, as in the Vulkan Guide's "Swapchain Semaphore Reuse" chapter), which
//!   `vkQueuePresentKHR` waits for.
//! - Resizes, `VK_SUBOPTIMAL_KHR` and `VK_ERROR_OUT_OF_DATE_KHR` recreate the swapchain at the
//!   next acquisition, passing the old one as `oldSwapchain`. Without
//!   `VK_EXT_swapchain_maintenance1` there is no fence for presentation, so the retired swapchain
//!   is destroyed once the GPU has completed the submission after its retirement.
//! - A surface has at most one swapchain that is not retired, and only `oldSwapchain` retires
//!   one: creating another with a null `oldSwapchain` fails with
//!   `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR` (ISSUE-012). So while the window has no area, the
//!   swapchain gives up its images but stays alive ("parked") until the next creation retires
//!   it.

use crate::device::{lock, Garbage, VulkanDevice};
use crate::resource::{TextureStorage, VulkanTexture};
use ash::vk;
use korppi_rhi::SmallVec;
use korppi_rhi::{
    PresentMode, PresentTarget, RhiError, SampleCount, SubmissionId, SwapchainDescriptor,
    SwapchainHandle, SwapchainImage, SwapchainInfo, TextureDimension, TextureFormat, TextureHandle,
    TextureInfo, TextureUsage,
};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// How long `acquire_next_image` waits for an image before reporting none (for example while a
/// compositor withholds images of a hidden window).
const ACQUIRE_TIMEOUT_NS: u64 = 1_000_000_000;

/// One image of a swapchain.
struct SwapchainSlot {
    texture: TextureHandle,
    image: vk::Image,
    view: vk::ImageView,
    /// Signalled by the present batch, waited for by `vkQueuePresentKHR`. Indexed by image: an
    /// image is only acquired again after its previous presentation has waited for it.
    present_semaphore: vk::Semaphore,
}

/// The image acquired and not yet presented.
struct Acquired {
    index: u32,
    /// The acquire semaphore, until a submission waits for it.
    unwaited: Option<vk::Semaphore>,
    /// Layout the submitted commands left the image in (`UNDEFINED` after acquisition).
    layout: vk::ImageLayout,
}

/// Binary semaphores for acquisitions. A semaphore is reused once the submission that waited
/// for it has completed.
#[derive(Default)]
struct SemaphorePool {
    free: Vec<vk::Semaphore>,
    waited: VecDeque<(SubmissionId, vk::Semaphore)>,
}

impl SemaphorePool {
    fn recycle(&mut self, completed: u64) {
        while let Some(&(submission, semaphore)) = self.waited.front() {
            if submission.get() > completed {
                break;
            }
            self.waited.pop_front();
            self.free.push(semaphore);
        }
    }

    fn take(&mut self, device: &ash::Device) -> Result<vk::Semaphore, RhiError> {
        if let Some(semaphore) = self.free.pop() {
            return Ok(semaphore);
        }
        create_binary_semaphore(device)
    }

    fn drain(&mut self) -> impl Iterator<Item = vk::Semaphore> + '_ {
        self.free
            .drain(..)
            .chain(self.waited.drain(..).map(|(_, semaphore)| semaphore))
    }
}

/// A window surface and its swapchain, owned by a `VulkanDevice`.
pub(crate) struct VulkanSwapchain {
    /// Keeps the window alive until the surface is destroyed.
    target: Arc<dyn PresentTarget>,
    surface: vk::SurfaceKHR,
    /// The swapchain whose images are registered; null while the window has no area.
    swapchain: vk::SwapchainKHR,
    /// While the window has no area, the swapchain that last had images: still the surface's
    /// non-retired swapchain, retired by passing it as `oldSwapchain` at the next creation.
    parked: vk::SwapchainKHR,
    slots: Vec<SwapchainSlot>,
    surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    info: SwapchainInfo,
    requested_extent: (u32, u32),
    requested_image_count: u32,
    needs_recreate: bool,
    acquired: Option<Acquired>,
    semaphores: SemaphorePool,
}

/// Swapchains of a device, each behind its own lock (acquisition may block).
pub(crate) type SwapchainMap = Mutex<std::collections::HashMap<u32, Arc<Mutex<VulkanSwapchain>>>>;

fn create_binary_semaphore(device: &ash::Device) -> Result<vk::Semaphore, RhiError> {
    // SAFETY: a default create info describes a binary semaphore; it is destroyed by the
    // swapchain code once no operation is pending on it.
    unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
        .map_err(|e| RhiError::OutOfMemory(format!("vkCreateSemaphore failed: {e}")))
}

/// The texture format of a swapchain format, for the formats the RHI exposes.
fn texture_format(format: vk::Format) -> Option<TextureFormat> {
    match format {
        vk::Format::B8G8R8A8_SRGB => Some(TextureFormat::Bgra8Srgb),
        vk::Format::R8G8B8A8_SRGB => Some(TextureFormat::Rgba8Srgb),
        vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
        vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        _ => None,
    }
}

/// Chooses an 8-bit BGRA/RGBA format in the sRGB colour space, preferring an sRGB-encoded
/// format if `srgb` (VUID-VkSwapchainCreateInfoKHR-imageFormat-01273: the pair must be one the
/// surface reports).
pub(crate) fn choose_surface_format(
    available: &[vk::SurfaceFormatKHR],
    srgb: bool,
) -> Option<vk::SurfaceFormatKHR> {
    let preferred: [vk::Format; 4] = if srgb {
        [
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
        ]
    } else {
        [
            vk::Format::B8G8R8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R8G8B8A8_SRGB,
        ]
    };
    preferred.into_iter().find_map(|format| {
        available
            .iter()
            .find(|f| f.format == format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .copied()
    })
}

fn to_vk_present_mode(mode: PresentMode) -> vk::PresentModeKHR {
    match mode {
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
    }
}

fn from_vk_present_mode(mode: vk::PresentModeKHR) -> PresentMode {
    match mode {
        vk::PresentModeKHR::FIFO_RELAXED => PresentMode::FifoRelaxed,
        vk::PresentModeKHR::MAILBOX => PresentMode::Mailbox,
        vk::PresentModeKHR::IMMEDIATE => PresentMode::Immediate,
        _ => PresentMode::Fifo,
    }
}

/// The requested mode if the surface supports it, otherwise FIFO (required to be supported).
pub(crate) fn choose_present_mode(
    available: &[vk::PresentModeKHR],
    requested: PresentMode,
) -> vk::PresentModeKHR {
    let mode = to_vk_present_mode(requested);
    if available.contains(&mode) {
        mode
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// `requested` (or one more than the minimum, if 0), within the surface's limits
/// (VUID-VkSwapchainCreateInfoKHR-presentMode-02839; `max_image_count == 0` means no maximum).
pub(crate) fn choose_image_count(caps: &vk::SurfaceCapabilitiesKHR, requested: u32) -> u32 {
    let count = if requested == 0 {
        caps.min_image_count + 1
    } else {
        requested.max(caps.min_image_count)
    };
    if caps.max_image_count > 0 {
        count.min(caps.max_image_count)
    } else {
        count
    }
}

/// The surface's current extent, or the requested one within its limits where the surface lets
/// the swapchain decide (`current_extent` of `0xFFFFFFFF`, as on Wayland).
pub(crate) fn choose_extent(
    caps: &vk::SurfaceCapabilitiesKHR,
    width: u32,
    height: u32,
) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        return caps.current_extent;
    }
    vk::Extent2D {
        width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
        height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    }
}

/// Opaque composition if supported, otherwise the first supported mode
/// (VUID-VkSwapchainCreateInfoKHR-compositeAlpha-01280).
pub(crate) fn choose_composite_alpha(
    supported: vk::CompositeAlphaFlagsKHR,
) -> vk::CompositeAlphaFlagsKHR {
    [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::INHERIT,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
    ]
    .into_iter()
    .find(|&mode| supported.contains(mode))
    .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE)
}

/// Identity if supported, otherwise the surface's current transform
/// (VUID-VkSwapchainCreateInfoKHR-preTransform-01279).
pub(crate) fn choose_transform(caps: &vk::SurfaceCapabilitiesKHR) -> vk::SurfaceTransformFlagsKHR {
    if caps
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        caps.current_transform
    }
}

/// Colour attachment (always supported), plus copy source/destination where supported.
fn choose_usage(supported: vk::ImageUsageFlags) -> (vk::ImageUsageFlags, TextureUsage) {
    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    let mut texture_usage = TextureUsage::COLOR_ATTACHMENT;
    if supported.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        usage |= vk::ImageUsageFlags::TRANSFER_DST;
        texture_usage |= TextureUsage::COPY_DST;
    }
    if supported.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        texture_usage |= TextureUsage::COPY_SRC;
    }
    (usage, texture_usage)
}

/// Native objects of a retired swapchain.
pub(crate) struct RetiredSwapchain {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) views: Vec<vk::ImageView>,
    pub(crate) semaphores: Vec<vk::Semaphore>,
}

impl RetiredSwapchain {
    /// A swapchain (possibly null) whose images were released earlier.
    fn bare(surface: vk::SurfaceKHR, swapchain: vk::SwapchainKHR) -> Self {
        Self {
            surface,
            swapchain,
            views: Vec::new(),
            semaphores: Vec::new(),
        }
    }
}

impl VulkanDevice {
    fn swapchain_loaders(
        &self,
    ) -> Result<(&ash::khr::surface::Instance, &ash::khr::swapchain::Device), RhiError> {
        match (
            self.context.surface_loader.as_ref(),
            self.context.swapchain_loader.as_ref(),
        ) {
            (Some(surface), Some(swapchain)) => Ok((surface, swapchain)),
            _ => Err(RhiError::Unsupported(
                "the Vulkan driver offers no window-surface or swapchain extension".into(),
            )),
        }
    }

    /// The id of a swapchain handle of this device; a handle of another device is rejected
    /// (RHI-15), since swapchains are kept by id, not in a slot map.
    fn swapchain_key(&self, handle: SwapchainHandle) -> Result<u32, RhiError> {
        if handle.device() == self.resources.device {
            Ok(handle.id())
        } else {
            Err(RhiError::InvalidHandle(format!(
                "swapchain {handle:?} belongs to another device"
            )))
        }
    }

    fn swapchain_entry(
        &self,
        handle: SwapchainHandle,
    ) -> Result<Arc<Mutex<VulkanSwapchain>>, RhiError> {
        let key = self.swapchain_key(handle)?;
        lock(&self.swapchains)
            .get(&key)
            .cloned()
            .ok_or_else(|| RhiError::InvalidHandle(format!("swapchain {handle:?} not found")))
    }

    pub(crate) fn create_swapchain_impl(
        &self,
        target: Arc<dyn PresentTarget>,
        desc: &SwapchainDescriptor,
    ) -> Result<SwapchainHandle, RhiError> {
        let (surface_loader, _) = self.swapchain_loaders()?;
        let display = target
            .display_handle()
            .map_err(|e| RhiError::SwapchainError(format!("no display handle: {e}")))?
            .as_raw();
        let window = target
            .window_handle()
            .map_err(|e| RhiError::SwapchainError(format!("no window handle: {e}")))?
            .as_raw();
        let required = ash_window::enumerate_required_extensions(display)
            .map_err(|e| RhiError::Unsupported(format!("unsupported window system: {e}")))?;
        for &extension in required {
            // SAFETY: `ash-window` returns pointers to static NUL-terminated extension names.
            let name = unsafe { std::ffi::CStr::from_ptr(extension) };
            if !self.context.surface_extensions.contains(&name) {
                return Err(RhiError::Unsupported(format!(
                    "the Vulkan loader does not offer {}",
                    name.to_string_lossy()
                )));
            }
        }
        // SAFETY: the handles come from a live window and display that `target` keeps alive
        // until the surface is destroyed (the swapchain owns `target`); the instance enables the
        // extensions the window system needs (checked above).
        let surface = unsafe {
            ash_window::create_surface(
                &self.context.entry,
                &self.context.instance,
                display,
                window,
                None,
            )
        }
        .map_err(|e| RhiError::SwapchainError(format!("surface creation failed: {e}")))?;

        let negotiated = (|| {
            // SAFETY: `surface` was created from this instance; the physical device and queue
            // family belong to the device.
            unsafe {
                let supported = surface_loader
                    .get_physical_device_surface_support(
                        self.context.physical_device,
                        self.context.graphics_queue_family,
                        surface,
                    )
                    .map_err(|e| RhiError::SwapchainError(format!("surface support query: {e}")))?;
                if !supported {
                    return Err(RhiError::Unsupported(
                        "the device's queue family cannot present to this window".into(),
                    ));
                }
                let formats = surface_loader
                    .get_physical_device_surface_formats(self.context.physical_device, surface)
                    .map_err(|e| RhiError::SwapchainError(format!("surface format query: {e}")))?;
                let modes = surface_loader
                    .get_physical_device_surface_present_modes(
                        self.context.physical_device,
                        surface,
                    )
                    .map_err(|e| RhiError::SwapchainError(format!("present mode query: {e}")))?;
                let surface_format =
                    choose_surface_format(&formats, desc.srgb).ok_or_else(|| {
                        RhiError::Unsupported(format!(
                            "no 8-bit BGRA/RGBA sRGB-colour-space surface format in {formats:?}"
                        ))
                    })?;
                Ok((
                    surface_format,
                    choose_present_mode(&modes, desc.present_mode),
                ))
            }
        })();
        let (surface_format, present_mode) = match negotiated {
            Ok(negotiated) => negotiated,
            Err(error) => {
                // SAFETY: no swapchain was created for the surface.
                unsafe { surface_loader.destroy_surface(surface, None) };
                return Err(error);
            }
        };
        let format = texture_format(surface_format.format).expect("chosen formats are mapped");

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let mut swapchain = VulkanSwapchain {
            target,
            surface,
            swapchain: vk::SwapchainKHR::null(),
            parked: vk::SwapchainKHR::null(),
            slots: Vec::new(),
            surface_format,
            present_mode,
            info: SwapchainInfo {
                format,
                usage: TextureUsage::COLOR_ATTACHMENT,
                width: 0,
                height: 0,
                present_mode: from_vk_present_mode(present_mode),
                image_count: 0,
            },
            requested_extent: (desc.width, desc.height),
            requested_image_count: desc.image_count,
            needs_recreate: true,
            acquired: None,
            semaphores: SemaphorePool::default(),
        };
        if let Err(error) = self.recreate(&mut swapchain, id) {
            self.destroy_swapchain_objects(swapchain);
            return Err(error);
        }
        lock(&self.swapchains).insert(id, Arc::new(Mutex::new(swapchain)));
        Ok(SwapchainHandle::from_parts(self.resources.device, id, 1))
    }

    /// (Re)creates the swapchain for the requested extent. Returns `false`, with no swapchain,
    /// while the window has no area.
    fn recreate(&self, sc: &mut VulkanSwapchain, id: u32) -> Result<bool, RhiError> {
        let (surface_loader, swapchain_loader) = self.swapchain_loaders()?;
        sc.needs_recreate = false;
        // SAFETY: the surface and physical device belong to this instance.
        let caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(self.context.physical_device, sc.surface)
        }
        .map_err(|e| RhiError::SwapchainError(format!("surface capability query: {e}")))?;
        let (width, height) = sc.requested_extent;
        let extent = choose_extent(&caps, width, height);
        if width == 0 || height == 0 || extent.width == 0 || extent.height == 0 {
            // Give up the images but keep the swapchain, which no creation has retired yet.
            let current = std::mem::replace(&mut sc.swapchain, vk::SwapchainKHR::null());
            self.retire(sc);
            if current != vk::SwapchainKHR::null() {
                sc.parked = current;
            }
            sc.info.width = 0;
            sc.info.height = 0;
            sc.info.image_count = 0;
            return Ok(false);
        }

        let (usage, texture_usage) = choose_usage(caps.supported_usage_flags);
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(sc.surface)
            .min_image_count(choose_image_count(&caps, sc.requested_image_count))
            .image_format(sc.surface_format.format)
            .image_color_space(sc.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(choose_transform(&caps))
            .composite_alpha(choose_composite_alpha(caps.supported_composite_alpha))
            .present_mode(sc.present_mode)
            .clipped(true)
            .old_swapchain(if sc.swapchain != vk::SwapchainKHR::null() {
                sc.swapchain
            } else {
                sc.parked
            });
        // SAFETY: every parameter was chosen from this surface's queries; `oldSwapchain` is the
        // surface's non-retired swapchain (or null if it has none) and has no acquired image
        // (acquisition requires the previous image to be presented).
        let created = unsafe { swapchain_loader.create_swapchain(&create_info, None) };
        // The call retires `oldSwapchain` whether or not it succeeds.
        self.retire(sc);
        let parked = std::mem::replace(&mut sc.parked, vk::SwapchainKHR::null());
        self.defer_retired(RetiredSwapchain::bare(sc.surface, parked));
        let swapchain = created.map_err(|e| {
            sc.needs_recreate = true;
            sc.info.width = 0;
            sc.info.height = 0;
            sc.info.image_count = 0;
            RhiError::SwapchainError(format!("vkCreateSwapchainKHR failed: {e}"))
        })?;
        sc.swapchain = swapchain;

        // SAFETY: `swapchain` was just created.
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.map_err(|e| {
            RhiError::SwapchainError(format!("vkGetSwapchainImagesKHR failed: {e}"))
        })?;
        let device = &self.context.device;
        for (index, image) in images.into_iter().enumerate() {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(sc.surface_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: `image` belongs to the live swapchain; the view is destroyed with it.
            let view = unsafe { device.create_image_view(&view_info, None) }
                .map_err(|e| RhiError::SwapchainError(format!("vkCreateImageView failed: {e}")))?;
            let present_semaphore = match create_binary_semaphore(device) {
                Ok(semaphore) => semaphore,
                Err(error) => {
                    // SAFETY: the view was just created and is not used yet.
                    unsafe { device.destroy_image_view(view, None) };
                    return Err(error);
                }
            };
            let (texture_id, generation) = lock(&self.resources.textures).insert(VulkanTexture {
                image,
                view,
                storage: TextureStorage::Swapchain {
                    swapchain: id,
                    index: index as u32,
                },
                format: sc.info.format,
                usage: texture_usage,
                layouts: vec![vk::ImageLayout::UNDEFINED],
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                info: TextureInfo {
                    width: extent.width,
                    height: extent.height,
                    format: sc.info.format,
                    dimension: TextureDimension::D2,
                    mip_levels: 1,
                    array_layers: 1,
                    depth: 1,
                    sample_count: SampleCount::Count1,
                    usage: texture_usage,
                },
                // The default view is single-mip, single-layer; the swapchain destroys it.
                attachment_views: vec![vec![view]],
                views: Vec::new(),
            });
            sc.slots.push(SwapchainSlot {
                texture: TextureHandle::from_parts(self.resources.device, texture_id, generation),
                image,
                view,
                present_semaphore,
            });
        }
        sc.info.usage = texture_usage;
        sc.info.width = extent.width;
        sc.info.height = extent.height;
        sc.info.image_count = sc.slots.len() as u32;
        Ok(true)
    }

    /// Unregisters the current images and hands them and the current swapchain to deferred
    /// destruction.
    fn retire(&self, sc: &mut VulkanSwapchain) {
        let retired = self.take_swapchain_objects(sc);
        self.defer_retired(retired);
    }

    /// Destroys `retired` once the submission after this point has completed (see the module
    /// documentation); does nothing if it holds no objects.
    fn defer_retired(&self, retired: RetiredSwapchain) {
        if retired.swapchain != vk::SwapchainKHR::null() || !retired.views.is_empty() {
            let after = SubmissionId::new(self.queue.last_issued().get() + 1);
            lock(&self.garbage).push_back((after, Garbage::Swapchain(retired)));
        }
    }

    /// Removes the swapchain's images from the registry and returns its native objects.
    fn take_swapchain_objects(&self, sc: &mut VulkanSwapchain) -> RetiredSwapchain {
        let mut textures = lock(&self.resources.textures);
        let mut retired = RetiredSwapchain {
            surface: sc.surface,
            swapchain: std::mem::replace(&mut sc.swapchain, vk::SwapchainKHR::null()),
            views: Vec::with_capacity(sc.slots.len()),
            semaphores: Vec::with_capacity(sc.slots.len()),
        };
        for slot in sc.slots.drain(..) {
            textures.remove(slot.texture);
            retired.views.push(slot.view);
            retired.semaphores.push(slot.present_semaphore);
        }
        retired
    }

    /// Destroys a retired swapchain's objects. The GPU and presentation engine must be done
    /// with them.
    pub(crate) fn destroy_retired(&self, retired: RetiredSwapchain) {
        let device = &self.context.device;
        // SAFETY: the caller guarantees no pending operation uses the objects; views are
        // destroyed before the swapchain that owns their images.
        unsafe {
            for view in retired.views {
                device.destroy_image_view(view, None);
            }
            if retired.swapchain != vk::SwapchainKHR::null() {
                if let Some(loader) = &self.context.swapchain_loader {
                    loader.destroy_swapchain(retired.swapchain, None);
                }
            }
            for semaphore in retired.semaphores {
                device.destroy_semaphore(semaphore, None);
            }
        }
    }

    /// Destroys a swapchain, its retired predecessors and its surface. Waits for the device to
    /// be idle first: presentation requests are queue operations no timeline value covers.
    fn destroy_swapchain_objects(&self, mut sc: VulkanSwapchain) {
        if let Err(error) = self.queue.wait_idle() {
            tracing::error!("waiting for the queue before destroying a swapchain: {error}");
        }
        // Retired swapchains of this surface must be destroyed before it.
        let retired: Vec<RetiredSwapchain> = {
            let mut garbage = lock(&self.garbage);
            let mut kept = VecDeque::with_capacity(garbage.len());
            let mut retired = Vec::new();
            for (after, item) in garbage.drain(..) {
                match item {
                    Garbage::Swapchain(r) if r.surface == sc.surface => retired.push(r),
                    other => kept.push_back((after, other)),
                }
            }
            *garbage = kept;
            retired
        };
        for r in retired {
            self.destroy_retired(r);
        }
        let mut current = self.take_swapchain_objects(&mut sc);
        current.semaphores.extend(sc.semaphores.drain());
        if let Some(acquired) = sc.acquired.take() {
            current.semaphores.extend(acquired.unwaited);
        }
        let surface = current.surface;
        self.destroy_retired(current);
        let parked = std::mem::replace(&mut sc.parked, vk::SwapchainKHR::null());
        self.destroy_retired(RetiredSwapchain::bare(surface, parked));
        if let Some(loader) = &self.context.surface_loader {
            // SAFETY: every swapchain created for the surface was destroyed above.
            unsafe { loader.destroy_surface(surface, None) };
        }
        // `sc.target` (the window) is released here, after its surface.
    }

    pub(crate) fn destroy_swapchain_impl(&self, handle: SwapchainHandle) -> Result<(), RhiError> {
        let entry = lock(&self.swapchains)
            .remove(&self.swapchain_key(handle)?)
            .ok_or_else(|| RhiError::InvalidHandle(format!("swapchain {handle:?} not found")))?;
        // Waits for a call on this swapchain that another thread may be inside of.
        let sc = VulkanSwapchain::take(&entry);
        drop(entry);
        self.destroy_swapchain_objects(sc);
        Ok(())
    }

    /// Destroys every swapchain (device teardown; the device is idle).
    pub(crate) fn destroy_all_swapchains(&self) {
        let all: Vec<_> = lock(&self.swapchains).drain().collect();
        for (_, entry) in all {
            let sc = VulkanSwapchain::take(&entry);
            self.destroy_swapchain_objects(sc);
        }
    }

    pub(crate) fn resize_swapchain_impl(
        &self,
        handle: SwapchainHandle,
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        let entry = self.swapchain_entry(handle)?;
        let mut sc = lock(&entry);
        if sc.requested_extent != (width, height)
            || (sc.info.width, sc.info.height) != (width, height)
        {
            sc.requested_extent = (width, height);
            sc.needs_recreate = true;
        }
        Ok(())
    }

    pub(crate) fn swapchain_info_impl(
        &self,
        handle: SwapchainHandle,
    ) -> Result<SwapchainInfo, RhiError> {
        let entry = self.swapchain_entry(handle)?;
        let sc = lock(&entry);
        Ok(sc.info)
    }

    pub(crate) fn acquire_impl(
        &self,
        handle: SwapchainHandle,
    ) -> Result<Option<SwapchainImage>, RhiError> {
        let (_, swapchain_loader) = self.swapchain_loaders()?;
        let entry = self.swapchain_entry(handle)?;
        let mut sc = lock(&entry);
        if sc.acquired.is_some() {
            return Err(RhiError::ValidationError(format!(
                "swapchain {handle:?}: the acquired image must be presented before acquiring another"
            )));
        }
        let completed = self.queue.completed()?;
        sc.semaphores.recycle(completed);

        // An out-of-date result recreates the swapchain and tries again; a surface that keeps
        // changing (an interactive resize) can make that happen more than once.
        for _ in 0..3 {
            if (sc.needs_recreate || sc.swapchain == vk::SwapchainKHR::null())
                && !self.recreate(&mut sc, handle.id())?
            {
                return Ok(None);
            }
            let semaphore = sc.semaphores.take(&self.context.device)?;
            // SAFETY: the swapchain is externally synchronized by its lock; `semaphore` is
            // unsignalled with no pending operation (fresh, or its waiting submission completed).
            let result = unsafe {
                swapchain_loader.acquire_next_image(
                    sc.swapchain,
                    ACQUIRE_TIMEOUT_NS,
                    semaphore,
                    vk::Fence::null(),
                )
            };
            match result {
                Ok((index, suboptimal)) => {
                    // A suboptimal image can still be presented; recreate afterwards.
                    sc.needs_recreate |= suboptimal;
                    sc.acquired = Some(Acquired {
                        index,
                        unwaited: Some(semaphore),
                        layout: vk::ImageLayout::UNDEFINED,
                    });
                    let slot = &sc.slots[index as usize];
                    return Ok(Some(SwapchainImage {
                        swapchain: handle,
                        index,
                        texture: slot.texture,
                        width: sc.info.width,
                        height: sc.info.height,
                    }));
                }
                // A failed acquisition leaves the semaphore untouched.
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    sc.semaphores.free.push(semaphore);
                    sc.needs_recreate = true;
                }
                Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                    sc.semaphores.free.push(semaphore);
                    tracing::debug!("swapchain {handle:?}: no image available; skipping a frame");
                    return Ok(None);
                }
                Err(vk::Result::ERROR_DEVICE_LOST) => {
                    sc.semaphores.free.push(semaphore);
                    return Err(crate::error::vk_error(
                        "vkAcquireNextImageKHR",
                        vk::Result::ERROR_DEVICE_LOST,
                    ));
                }
                Err(error) => {
                    sc.semaphores.free.push(semaphore);
                    return Err(RhiError::SwapchainError(format!(
                        "vkAcquireNextImageKHR failed: {error}"
                    )));
                }
            }
        }
        Err(RhiError::SwapchainError(
            "the swapchain stayed out of date after recreation".into(),
        ))
    }

    pub(crate) fn present_impl(&self, image: &SwapchainImage) -> Result<(), RhiError> {
        let (_, swapchain_loader) = self.swapchain_loaders()?;
        let entry = self.swapchain_entry(image.swapchain)?;
        let mut sc = lock(&entry);
        let is_acquired = sc.acquired.as_ref().is_some_and(|a| a.index == image.index)
            && sc
                .slots
                .get(image.index as usize)
                .is_some_and(|slot| slot.texture == image.texture);
        if !is_acquired {
            return Err(RhiError::ValidationError(format!(
                "swapchain {:?} image {} is not acquired (already presented, or from before a \
                 recreation)",
                image.swapchain, image.index
            )));
        }
        let (layout, unwaited) = {
            let acquired = sc.acquired.as_ref().expect("checked above");
            (acquired.layout, acquired.unwaited)
        };
        let slot_image = sc.slots[image.index as usize].image;
        let present_semaphore = sc.slots[image.index as usize].present_semaphore;

        // Transition to PRESENT_SRC_KHR unless the caller's commands already did.
        let recording = if layout == vk::ImageLayout::PRESENT_SRC_KHR {
            None
        } else {
            let recording = self.recording.acquire()?;
            let cmd = recording.command_buffer();
            let to_present = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(slot_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let device = &self.context.device;
            // SAFETY: the recording context is owned here and its command buffer is in the
            // initial state; the image is acquired and its layout is the one tracked from the
            // submitted commands (UNDEFINED discards the contents, which were never written).
            let recorded = unsafe {
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .and_then(|()| {
                        device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default()
                                .image_memory_barriers(std::slice::from_ref(&to_present)),
                        );
                        device.end_command_buffer(cmd)
                    })
            };
            if let Err(error) = recorded {
                self.recording.release(recording);
                return Err(RhiError::BackendError(format!(
                    "recording the present transition failed: {error}"
                )));
            }
            Some(recording)
        };

        let command_buffers: Vec<vk::CommandBuffer> =
            recording.iter().map(|r| r.command_buffer()).collect();
        // The barrier's source scope is ALL_COMMANDS, so waiting there chains the transition
        // after the acquisition.
        let waits: Vec<(vk::Semaphore, vk::PipelineStageFlags2)> = unwaited
            .map(|s| (s, vk::PipelineStageFlags2::ALL_COMMANDS))
            .into_iter()
            .collect();
        let submission =
            match self
                .queue
                .submit_batch(&command_buffers, &waits, &[present_semaphore])
            {
                Ok(submission) => submission,
                Err(error) => {
                    if let Some(recording) = recording {
                        self.recording.release(recording);
                    }
                    return Err(error);
                }
            };
        if let Some(recording) = recording {
            self.recording.retire(recording, submission);
        }
        self.frames.record(submission);
        if let Some(semaphore) = unwaited {
            sc.semaphores.waited.push_back((submission, semaphore));
        }
        sc.acquired = None;

        let swapchain = sc.swapchain;
        let presented =
            self.queue
                .present(swapchain_loader, swapchain, image.index, present_semaphore);
        let result = match presented {
            Ok(suboptimal) => {
                sc.needs_recreate |= suboptimal;
                Ok(())
            }
            // The presentation request was still queued (its semaphore wait executes); recreate
            // at the next acquisition.
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                sc.needs_recreate = true;
                Ok(())
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => Err(crate::error::vk_error(
                "vkQueuePresentKHR",
                vk::Result::ERROR_DEVICE_LOST,
            )),
            Err(error) => Err(RhiError::SwapchainError(format!(
                "vkQueuePresentKHR failed: {error}"
            ))),
        };
        drop(sc);
        self.collect_completed()?;
        result
    }

    /// For `submit`: checks the swapchain images `uses` refers to are acquired, and returns the
    /// acquire semaphores the batch must wait for.
    pub(crate) fn swapchain_waits(
        &self,
        uses: &[crate::command::SwapchainUse],
    ) -> Result<SmallVec<[(vk::Semaphore, vk::PipelineStageFlags2); 2]>, RhiError> {
        let mut waits = SmallVec::new();
        for used in uses {
            let entry = self.swapchain_entry(SwapchainHandle::from_parts(
                self.resources.device,
                used.swapchain,
                1,
            ))?;
            let sc = lock(&entry);
            let acquired = sc.acquired.as_ref().filter(|a| {
                a.index == used.index
                    && sc
                        .slots
                        .get(used.index as usize)
                        .is_some_and(|slot| slot.texture.id() == used.texture)
            });
            match acquired {
                Some(acquired) => waits.extend(
                    acquired
                        .unwaited
                        .map(|s| (s, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)),
                ),
                None => {
                    return Err(RhiError::ValidationError(format!(
                        "the command buffer uses image {} of swapchain {}, which is not acquired",
                        used.index, used.swapchain
                    )))
                }
            }
        }
        Ok(waits)
    }

    /// For `submit`, after the batch was queued: the acquire semaphores it waited for return to
    /// the pool once `submission` completes, and the images take the layouts the commands left.
    pub(crate) fn swapchains_submitted(
        &self,
        uses: &[crate::command::SwapchainUse],
        submission: SubmissionId,
    ) {
        for used in uses {
            let Ok(entry) = self.swapchain_entry(SwapchainHandle::from_parts(
                self.resources.device,
                used.swapchain,
                1,
            )) else {
                continue;
            };
            let mut sc = lock(&entry);
            let sc = &mut *sc;
            if let Some(acquired) = sc.acquired.as_mut().filter(|a| a.index == used.index) {
                if let Some(semaphore) = acquired.unwaited.take() {
                    sc.semaphores.waited.push_back((submission, semaphore));
                }
                if let Some(layout) = used.final_layout {
                    acquired.layout = layout;
                }
            }
        }
    }
}

impl VulkanSwapchain {
    /// Moves the swapchain out of a shared entry for destruction, leaving an inert stand-in
    /// that owns no Vulkan objects.
    fn take(entry: &Mutex<VulkanSwapchain>) -> Self {
        let mut guard = lock(entry);
        let placeholder = Self::placeholder(&guard);
        std::mem::replace(&mut *guard, placeholder)
    }

    fn placeholder(from: &VulkanSwapchain) -> Self {
        Self {
            target: Arc::clone(&from.target),
            surface: vk::SurfaceKHR::null(),
            swapchain: vk::SwapchainKHR::null(),
            parked: vk::SwapchainKHR::null(),
            slots: Vec::new(),
            surface_format: from.surface_format,
            present_mode: from.present_mode,
            info: from.info,
            requested_extent: (0, 0),
            requested_image_count: 0,
            needs_recreate: false,
            acquired: None,
            semaphores: SemaphorePool::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(current: (u32, u32), min_count: u32, max_count: u32) -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            min_image_count: min_count,
            max_image_count: max_count,
            current_extent: vk::Extent2D {
                width: current.0,
                height: current.1,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            supported_transforms: vk::SurfaceTransformFlagsKHR::IDENTITY,
            current_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            ..Default::default()
        }
    }

    fn format(format: vk::Format) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }
    }

    #[test]
    fn surface_format_prefers_srgb_and_respects_the_reported_pairs() {
        let available = [
            format(vk::Format::B8G8R8A8_UNORM),
            format(vk::Format::B8G8R8A8_SRGB),
        ];
        assert_eq!(
            choose_surface_format(&available, true),
            Some(format(vk::Format::B8G8R8A8_SRGB))
        );
        assert_eq!(
            choose_surface_format(&available, false),
            Some(format(vk::Format::B8G8R8A8_UNORM))
        );
        // Only sRGB-colour-space pairs the surface reports are eligible.
        let hdr = [vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        }];
        assert_eq!(choose_surface_format(&hdr, true), None);
        // Falls back to whichever 8-bit format exists.
        assert_eq!(
            choose_surface_format(&[format(vk::Format::R8G8B8A8_UNORM)], true),
            Some(format(vk::Format::R8G8B8A8_UNORM))
        );
    }

    #[test]
    fn present_mode_falls_back_to_fifo() {
        let available = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE];
        assert_eq!(
            choose_present_mode(&available, PresentMode::Immediate),
            vk::PresentModeKHR::IMMEDIATE
        );
        assert_eq!(
            choose_present_mode(&available, PresentMode::Mailbox),
            vk::PresentModeKHR::FIFO
        );
        assert_eq!(
            from_vk_present_mode(choose_present_mode(&available, PresentMode::FifoRelaxed)),
            PresentMode::Fifo
        );
    }

    #[test]
    fn image_count_respects_the_surface_limits() {
        assert_eq!(choose_image_count(&caps((64, 64), 2, 8), 0), 3);
        assert_eq!(choose_image_count(&caps((64, 64), 3, 0), 0), 4);
        assert_eq!(choose_image_count(&caps((64, 64), 2, 3), 6), 3);
        assert_eq!(choose_image_count(&caps((64, 64), 3, 8), 1), 3);
    }

    #[test]
    fn extent_follows_the_surface_unless_the_swapchain_decides() {
        assert_eq!(
            choose_extent(&caps((640, 480), 2, 0), 100, 100),
            vk::Extent2D {
                width: 640,
                height: 480
            }
        );
        assert_eq!(
            choose_extent(&caps((u32::MAX, u32::MAX), 2, 0), 800, 9000),
            vk::Extent2D {
                width: 800,
                height: 4096
            }
        );
    }

    #[test]
    fn composite_alpha_and_transform_come_from_the_supported_sets() {
        assert_eq!(
            choose_composite_alpha(
                vk::CompositeAlphaFlagsKHR::OPAQUE | vk::CompositeAlphaFlagsKHR::INHERIT
            ),
            vk::CompositeAlphaFlagsKHR::OPAQUE
        );
        assert_eq!(
            choose_composite_alpha(vk::CompositeAlphaFlagsKHR::INHERIT),
            vk::CompositeAlphaFlagsKHR::INHERIT
        );
        let mut rotated = caps((64, 64), 2, 0);
        rotated.supported_transforms = vk::SurfaceTransformFlagsKHR::ROTATE_90;
        rotated.current_transform = vk::SurfaceTransformFlagsKHR::ROTATE_90;
        assert_eq!(
            choose_transform(&rotated),
            vk::SurfaceTransformFlagsKHR::ROTATE_90
        );
    }

    #[test]
    fn usage_always_includes_colour_attachment() {
        let (usage, texture_usage) = choose_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);
        assert_eq!(usage, vk::ImageUsageFlags::COLOR_ATTACHMENT);
        assert_eq!(texture_usage, TextureUsage::COLOR_ATTACHMENT);
        let (usage, texture_usage) = choose_usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        );
        assert!(usage.contains(vk::ImageUsageFlags::TRANSFER_DST));
        assert!(texture_usage.contains(TextureUsage::COPY_SRC | TextureUsage::COPY_DST));
    }
}
