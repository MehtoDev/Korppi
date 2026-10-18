//! A Vulkan device on the process-wide instance: the physical device chosen, the logical device
//! and its graphics queue, and the device's validation messages.

use crate::capabilities::VulkanCapabilities;
use crate::instance::SharedInstance;
use ash::vk;
use ash::{Device, Entry, Instance};
use korppi_rhi::{DiagnosticLog, RhiError};
use std::ffi::CStr;
use std::sync::Arc;

pub use crate::instance::{SYNC_VALIDATION_ENV_VAR, VALIDATION_LAYER_NAME};

/// Core Vulkan 1.3 physical and logical device context.
pub struct VulkanContext {
    /// Vulkan loader entry points (of the process-wide instance).
    pub entry: Entry,
    /// The process-wide Vulkan instance, which outlives every device created from it.
    pub instance: Instance,
    /// Selected physical GPU.
    pub physical_device: vk::PhysicalDevice,
    /// Logical device handle.
    pub device: Device,
    /// Primary graphics queue.
    pub graphics_queue: vk::Queue,
    /// Graphics queue family index.
    pub graphics_queue_family: u32,
    /// Physical device memory heap and type properties.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Physical device hardware limits.
    pub limits: vk::PhysicalDeviceLimits,
    /// Largest single `vkAllocateMemory` (Vulkan 1.1 `maxMemoryAllocationSize`).
    pub max_memory_allocation_size: u64,
    /// Whether `VK_EXT_memory_budget` is enabled (heap budgets from the driver, RVK-08).
    pub memory_budget_enabled: bool,
    /// Whether `VK_LAYER_KHRONOS_validation` is enabled on the instance.
    pub validation_enabled: bool,
    /// Whether the `samplerAnisotropy` device feature is enabled.
    pub sampler_anisotropy: bool,
    /// Validation messages about this device, received through the instance's
    /// `VK_EXT_debug_utils` messenger if one is installed.
    pub diagnostics: Arc<DiagnosticLog>,
    /// Adapter information, limits, features and format support (RHI-09).
    pub capabilities: VulkanCapabilities,
    /// Window-surface instance extensions enabled on the instance (`VK_KHR_surface` first).
    pub surface_extensions: Vec<&'static CStr>,
    /// `VK_KHR_surface` functions; `None` if the loader offers no window-surface extensions.
    pub surface_loader: Option<ash::khr::surface::Instance>,
    /// `VK_KHR_swapchain` functions; `None` if the device cannot present.
    pub swapchain_loader: Option<ash::khr::swapchain::Device>,
    /// `VK_EXT_debug_utils` device functions (object names, command labels); `None` if the
    /// loader does not offer the extension.
    pub debug_utils: Option<ash::ext::debug_utils::Device>,
    /// The device's hold on the instance; released after the device is destroyed.
    shared: SharedInstance,
}

/// Physical/logical device state created after the instance.
struct DeviceSelection {
    capabilities: VulkanCapabilities,
    physical_device: vk::PhysicalDevice,
    swapchain_enabled: bool,
    sampler_anisotropy: bool,
    device: Device,
    graphics_queue: vk::Queue,
    graphics_queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    limits: vk::PhysicalDeviceLimits,
    max_memory_allocation_size: u64,
    memory_budget_enabled: bool,
}

impl VulkanContext {
    /// Creates a Vulkan 1.3 device that needs no window system (offscreen rendering, tests, CI),
    /// on the process-wide instance (created by the first device; see [`crate::instance`]).
    ///
    /// When the loader and driver offer them, the window-surface instance extensions and
    /// `VK_KHR_swapchain` are enabled as well, so the context can also present to windows of any
    /// supported window system ([`VulkanContext::swapchain_loader`]).
    ///
    /// When API validation is requested (debug builds, or `KORPPI_RHI_VALIDATION=1`) when the
    /// instance is created, the Khronos validation layer and a `VK_EXT_debug_utils` messenger
    /// are enabled if the system provides them; messages about this device are logged through
    /// `tracing` and collected in [`VulkanContext::diagnostics`].
    pub fn create_headless() -> Result<Arc<Self>, RhiError> {
        let shared = SharedInstance::acquire()?;
        let selection =
            select_and_create_device(&shared.instance, !shared.surface_extensions.is_empty())?;
        let diagnostics = Arc::new(DiagnosticLog::new());
        if let Some(router) = shared.diagnostic_router() {
            router.register_device(selection.device.handle(), &diagnostics);
            router.register_object(selection.graphics_queue, &diagnostics);
        }
        let swapchain_loader = selection
            .swapchain_enabled
            .then(|| ash::khr::swapchain::Device::new(&shared.instance, &selection.device));
        let debug_utils = shared
            .debug_utils_enabled
            .then(|| ash::ext::debug_utils::Device::new(&shared.instance, &selection.device));
        Ok(Arc::new(Self {
            capabilities: selection.capabilities,
            surface_extensions: shared.surface_extensions.clone(),
            surface_loader: shared.surface_loader.clone(),
            swapchain_loader,
            debug_utils,
            entry: shared.entry.clone(),
            instance: shared.instance.clone(),
            physical_device: selection.physical_device,
            device: selection.device,
            graphics_queue: selection.graphics_queue,
            graphics_queue_family: selection.graphics_queue_family,
            memory_properties: selection.memory_properties,
            limits: selection.limits,
            max_memory_allocation_size: selection.max_memory_allocation_size,
            memory_budget_enabled: selection.memory_budget_enabled,
            sampler_anisotropy: selection.sampler_anisotropy,
            validation_enabled: shared.validation_enabled,
            diagnostics,
            shared,
        }))
    }

    /// Whether a `VK_EXT_debug_utils` messenger feeds [`VulkanContext::diagnostics`].
    pub fn debug_messenger_installed(&self) -> bool {
        self.shared.diagnostic_router().is_some()
    }

    /// Routes validation messages about `command_buffer` to this device's diagnostics.
    pub(crate) fn register_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        if let Some(router) = self.shared.diagnostic_router() {
            router.register_object(command_buffer, &self.diagnostics);
        }
    }

    /// Stops routing messages about `command_buffer`; call it after the buffer is freed.
    pub(crate) fn unregister_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        if let Some(router) = self.shared.diagnostic_router() {
            router.unregister_object(command_buffer, &self.diagnostics);
        }
    }

    /// Finds a memory type index matching the specified filter and property flags.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, RhiError> {
        for (i, mem_type) in self.memory_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) != 0 && mem_type.property_flags.contains(properties) {
                return Ok(i as u32);
            }
        }
        Err(RhiError::OutOfMemory(format!(
            "Failed to find suitable Vulkan memory type for filter {:#x} and flags {:?}",
            type_filter, properties
        )))
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: `VulkanContext` is the last owner of the device (every other wrapper holds an
        // `Arc<VulkanContext>`), so no child object is still in use. The instance outlives the
        // device: `self.shared` is released after this function returns.
        unsafe { self.device.destroy_device(None) };
        if let Some(router) = self.shared.diagnostic_router() {
            router.unregister_object(self.graphics_queue, &self.diagnostics);
            router.unregister_device(self.device.handle(), &self.diagnostics);
        }
    }
}

/// Environment variable choosing the Vulkan device: an index into the enumeration order or a
/// case-insensitive substring of the device name. Without it the highest-scoring suitable device
/// is used (discrete > integrated > virtual > CPU).
pub const DEVICE_ENV_VAR: &str = "KORPPI_VK_DEVICE";

/// A physical device that meets the backend's requirements.
struct Candidate {
    physical_device: vk::PhysicalDevice,
    index: usize,
    name: String,
    score: u32,
    queue_family: u32,
    timestamp_valid_bits: u32,
}

/// Why `physical_device` cannot run the backend, or `None` if it can (RVK-14): Vulkan 1.3, the
/// dynamic-rendering, synchronization2 and timeline-semaphore features, and a queue family with
/// graphics and compute.
///
/// # Safety
/// `physical_device` must belong to `instance`.
unsafe fn unsuitable(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(u32, u32), String> {
    // SAFETY: forwarded from the caller.
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    if props.api_version < vk::API_VERSION_1_3 {
        return Err(format!(
            "Vulkan {}.{} (1.3 required)",
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version)
        ));
    }
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut features13)
        .push_next(&mut features12);
    // SAFETY: forwarded from the caller; the chain outlives the call.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
    let missing: Vec<&str> = [
        (features13.dynamic_rendering, "dynamicRendering"),
        (features13.synchronization2, "synchronization2"),
        (features12.timeline_semaphore, "timelineSemaphore"),
    ]
    .into_iter()
    .filter(|(supported, _)| *supported != vk::TRUE)
    .map(|(_, name)| name)
    .collect();
    if !missing.is_empty() {
        return Err(format!("missing features {missing:?}"));
    }
    // SAFETY: forwarded from the caller.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    families
        .iter()
        .position(|family| {
            family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        })
        .map(|family| (family as u32, families[family].timestamp_valid_bits))
        .ok_or_else(|| "no graphics + compute queue family".to_owned())
}

/// Picks the device to use among the suitable candidates.
fn choose_device(candidates: Vec<Candidate>, preference: Option<&str>) -> Option<Candidate> {
    if let Some(preference) = preference {
        let wanted = preference.trim().to_ascii_lowercase();
        let index = wanted.parse::<usize>().ok();
        let chosen = candidates
            .into_iter()
            .find(|c| Some(c.index) == index || c.name.to_ascii_lowercase().contains(&wanted));
        if chosen.is_none() {
            tracing::warn!("{DEVICE_ENV_VAR}={preference} matches no suitable Vulkan device");
        }
        return chosen;
    }
    candidates
        .into_iter()
        .max_by_key(|c| (c.score, std::cmp::Reverse(c.index)))
}

fn device_score(kind: vk::PhysicalDeviceType) -> u32 {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

fn select_and_create_device(
    instance: &Instance,
    surfaces_enabled: bool,
) -> Result<DeviceSelection, RhiError> {
    // SAFETY: `instance` is a valid, live instance for every call in this function.
    let physical_devices = unsafe {
        instance.enumerate_physical_devices().map_err(|e| {
            RhiError::InitializationFailed(format!("Failed to enumerate physical devices: {e}"))
        })?
    };

    if physical_devices.is_empty() {
        return Err(RhiError::InitializationFailed(
            "No Vulkan physical devices found".into(),
        ));
    }

    // Device scoring and requirement checks (RVK-14): unsuitable devices are skipped instead of
    // failing device creation.
    let mut candidates = Vec::new();
    for (index, &pdev) in physical_devices.iter().enumerate() {
        // SAFETY: `pdev` was returned by `enumerate_physical_devices` on this instance.
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        let name = props
            .device_name_as_c_str()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        // SAFETY: as above.
        match unsafe { unsuitable(instance, pdev) } {
            Ok((queue_family, timestamp_valid_bits)) => candidates.push(Candidate {
                physical_device: pdev,
                index,
                name,
                score: device_score(props.device_type),
                queue_family,
                timestamp_valid_bits,
            }),
            Err(reason) => tracing::info!("skipping Vulkan device {index} ({name}): {reason}"),
        }
    }
    let preference = std::env::var(DEVICE_ENV_VAR).ok();
    let candidate = choose_device(candidates, preference.as_deref()).ok_or_else(|| {
        RhiError::InitializationFailed(
            "No Vulkan 1.3 device with dynamic rendering, synchronization2, timeline semaphores \
             and a graphics + compute queue found"
                .into(),
        )
    })?;
    tracing::info!(
        "using Vulkan device {} ({})",
        candidate.index,
        candidate.name
    );
    let physical_device = candidate.physical_device;
    let graphics_queue_family = candidate.queue_family;
    // SAFETY: `physical_device` belongs to `instance`.
    let props = unsafe { instance.get_physical_device_properties(physical_device) };

    // SAFETY: `physical_device` belongs to `instance`.
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    // Logical device with Vulkan 1.3 core features.
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(graphics_queue_family)
        .queue_priorities(&queue_priorities);

    // Vulkan 1.3 Features: Dynamic Rendering & Synchronization2
    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    // Vulkan 1.2 Features: Timeline Semaphore
    let mut vulkan_12_features =
        vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);

    // Optional core features the RHI exposes, enabled when supported (RVK-05, RHI-09).
    // SAFETY: `physical_device` belongs to `instance`.
    let supported = unsafe { instance.get_physical_device_features(physical_device) };
    let core_features = crate::capabilities::features_to_enable(&supported);
    let sampler_anisotropy = core_features.sampler_anisotropy == vk::TRUE;

    // SAFETY: `physical_device` belongs to `instance`.
    let device_extensions =
        unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .unwrap_or_default();
    let offered = |name: &CStr| {
        device_extensions
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(name))
    };
    // Presentation (RVK-02): `VK_KHR_swapchain` when the instance can create window surfaces.
    let swapchain_enabled = surfaces_enabled && offered(ash::khr::swapchain::NAME);
    let mut enabled_extensions: Vec<*const std::ffi::c_char> = Vec::new();
    if swapchain_enabled {
        enabled_extensions.push(ash::khr::swapchain::NAME.as_ptr());
    }
    // Portability implementations (MoltenVK on macOS) must enable this extension (RVK-14).
    if offered(ash::khr::portability_subset::NAME) {
        enabled_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
    }
    // Heap budgets for the memory manager (RVK-08).
    let memory_budget_enabled = offered(ash::ext::memory_budget::NAME);
    if memory_budget_enabled {
        enabled_extensions.push(ash::ext::memory_budget::NAME.as_ptr());
    }
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    // SAFETY: `physical_device` belongs to `instance`, a Vulkan 1.3 instance.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
    let max_memory_allocation_size = maintenance3.max_memory_allocation_size;

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(&enabled_extensions)
        .enabled_features(&core_features)
        .push_next(&mut vulkan_13_features)
        .push_next(&mut vulkan_12_features);

    // SAFETY: `physical_device` belongs to `instance`; the create info and its `pNext` chain
    // outlive the call.
    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|e| {
                RhiError::InitializationFailed(format!(
                    "Failed to create Vulkan logical device: {e}"
                ))
            })?
    };

    // SAFETY: queue 0 of `graphics_queue_family` was requested in `device_create_info`.
    let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

    // SAFETY: `physical_device` belongs to `instance`.
    let capabilities = unsafe {
        crate::capabilities::query(
            instance,
            physical_device,
            &core_features,
            candidate.timestamp_valid_bits,
            swapchain_enabled,
        )
    };

    Ok(DeviceSelection {
        physical_device,
        swapchain_enabled,
        sampler_anisotropy,
        device,
        graphics_queue,
        graphics_queue_family,
        memory_properties,
        limits: props.limits,
        max_memory_allocation_size,
        memory_budget_enabled,
        capabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(index: usize, name: &str, kind: vk::PhysicalDeviceType) -> Candidate {
        Candidate {
            physical_device: vk::PhysicalDevice::null(),
            index,
            name: name.into(),
            score: device_score(kind),
            queue_family: 0,
            timestamp_valid_bits: 0,
        }
    }

    fn candidates() -> Vec<Candidate> {
        vec![
            candidate(0, "llvmpipe (LLVM 23)", vk::PhysicalDeviceType::CPU),
            candidate(
                1,
                "Intel(R) UHD Graphics",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
            ),
            candidate(
                2,
                "AMD Radeon RX 7800",
                vk::PhysicalDeviceType::DISCRETE_GPU,
            ),
        ]
    }

    #[test]
    fn devices_are_ranked_discrete_integrated_virtual_cpu() {
        assert_eq!(choose_device(candidates(), None).map(|c| c.index), Some(2));
        let mut no_discrete = candidates();
        no_discrete.pop();
        assert_eq!(choose_device(no_discrete, None).map(|c| c.index), Some(1));
        assert_eq!(choose_device(Vec::new(), None).map(|c| c.index), None);
    }

    #[test]
    fn the_device_can_be_chosen_by_index_or_name() {
        assert_eq!(
            choose_device(candidates(), Some("0")).map(|c| c.index),
            Some(0)
        );
        assert_eq!(
            choose_device(candidates(), Some("uhd")).map(|c| c.index),
            Some(1)
        );
        assert_eq!(
            choose_device(candidates(), Some("nvidia")).map(|c| c.index),
            None
        );
    }
}
