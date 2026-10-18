//! The process-wide Vulkan instance (ISSUE-015, ISSUE-016).
//!
//! Every [`VulkanContext`](crate::VulkanContext) in a process uses one `VkInstance`: the first
//! device creates it, and the last device to be destroyed destroys it. No instance is therefore
//! destroyed while another device in the process presents (on the owner's machine, destroying an
//! instance that presented to a Wayland surface crashed another instance's presentation), and the
//! loader, layers and drivers are initialized once.
//!
//! The drivers stay loaded until the process exits: before its first other call, the backend
//! creates a bare instance that it never destroys (see [`pin_drivers`]). The loader unloads a
//! driver when no instance needs it any more, and on the owner's machine unloading NVIDIA's
//! Vulkan driver corrupted the heap of its OpenGL driver, in use in the same process.
//!
//! Inputs: the loader's instance extensions and layers, and the validation environment
//! variables (`KORPPI_RHI_VALIDATION`, [`SYNC_VALIDATION_ENV_VAR`]), read when the instance is
//! created. Outputs: the instance and its window-surface functions ([`VulkanInstance`]), and the
//! validation messages, routed by [`DiagnosticRouter`] to the [`DiagnosticLog`] of the device
//! that caused them.

use ash::vk::{self, Handle};
use ash::{Entry, Instance};
use korppi_rhi::{api_validation_requested, DiagnosticLog, DiagnosticSeverity, RhiError};
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Name of the Khronos validation layer.
pub const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Environment variable that turns synchronization validation off (`0`) while the validation
/// layer is active; it is on by default.
pub const SYNC_VALIDATION_ENV_VAR: &str = "KORPPI_RHI_SYNC_VALIDATION";

/// `VK_EXT_layer_settings`, through which the validation layer's settings are configured.
const LAYER_SETTINGS_EXTENSION: &CStr = c"VK_EXT_layer_settings";

/// Platform window-surface instance extensions, enabled together with `VK_KHR_surface` when the
/// loader offers them so that a device can present to any window system it runs on (RVK-02).
const PLATFORM_SURFACE_EXTENSIONS: [&CStr; 6] = [
    ash::khr::xlib_surface::NAME,
    ash::khr::xcb_surface::NAME,
    ash::khr::wayland_surface::NAME,
    ash::khr::win32_surface::NAME,
    ash::ext::metal_surface::NAME,
    ash::khr::android_surface::NAME,
];

/// The instance every device in the process shares, while one exists.
static SHARED: Mutex<Option<Arc<VulkanInstance>>> = Mutex::new(None);

/// The bare instance that keeps the drivers loaded, if it could be created (see [`pin_drivers`]).
static PIN: OnceLock<Option<(Entry, Instance)>> = OnceLock::new();

/// Keeps the Vulkan drivers loaded until the process exits (ISSUE-016).
///
/// The loader loads the drivers for every instance and for every call that enumerates instance
/// layers or extensions while no instance exists, and unloads them again afterwards. Unloading
/// NVIDIA's Vulkan driver while its OpenGL driver is in use in the same process corrupts the
/// OpenGL driver's heap (loading it does not). A bare instance (no layers, no extensions), created
/// before anything else and never destroyed, keeps every driver loaded. If it cannot be created,
/// creating the real instance reports why.
///
/// Cost: one idle instance per process, created with the first device; its memory and creation
/// time are to be watched (known issues).
fn pin_drivers(entry: &Entry) {
    PIN.get_or_init(|| {
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: `create_info` and the application info it points to outlive the call.
        match unsafe { entry.create_instance(&create_info, None) } {
            Ok(instance) => Some((entry.clone(), instance)),
            Err(error) => {
                tracing::debug!("no Vulkan instance to keep the drivers loaded: {error}");
                None
            }
        }
    });
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A Vulkan instance with the window-surface extensions the loader offers and, when requested
/// and installed, the Khronos validation layer and a `VK_EXT_debug_utils` messenger.
pub struct VulkanInstance {
    /// Vulkan loader entry points.
    pub entry: Entry,
    /// Vulkan instance handle.
    pub instance: Instance,
    /// Window-surface instance extensions enabled on the instance (`VK_KHR_surface` first).
    pub surface_extensions: Vec<&'static CStr>,
    /// `VK_KHR_surface` functions; `None` if the loader offers no window-surface extensions.
    pub surface_loader: Option<ash::khr::surface::Instance>,
    /// Whether `VK_LAYER_KHRONOS_validation` is enabled on the instance.
    pub validation_enabled: bool,
    /// Whether `VK_EXT_debug_utils` is enabled on the instance.
    pub debug_utils_enabled: bool,
    /// Where the messenger's user data points; boxed so that its address is stable.
    router: Box<DiagnosticRouter>,
    debug_messenger: Option<DebugMessenger>,
}

/// A `VK_EXT_debug_utils` messenger and the loader used to destroy it.
struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanInstance {
    /// Creates a Vulkan 1.3 instance that needs no window system.
    ///
    /// When the loader offers them, the window-surface instance extensions are enabled as well.
    /// When API validation is requested (debug builds, or `KORPPI_RHI_VALIDATION=1`), the
    /// Khronos validation layer and a `VK_EXT_debug_utils` messenger are enabled if the system
    /// provides them; their messages are logged through `tracing` and routed to the device they
    /// concern ([`DiagnosticRouter`]).
    fn create() -> Result<Self, RhiError> {
        // SAFETY: loading the system Vulkan loader library; no Vulkan objects exist yet.
        let entry = unsafe {
            Entry::load().map_err(|e| {
                RhiError::InitializationFailed(format!("Failed to load Vulkan library: {e}"))
            })?
        };
        // Before the enumerations below, which would load and unload the drivers.
        pin_drivers(&entry);

        let validation_requested = api_validation_requested();
        let validation_enabled = validation_requested && layer_available(&entry);
        // Debug utils carry object names and command labels to debuggers and profilers, so they
        // are enabled whenever offered; the validation messenger only when validation is.
        let debug_utils_enabled = debug_utils_available(&entry, None)
            || (validation_enabled && debug_utils_available(&entry, Some(VALIDATION_LAYER_NAME)));
        if validation_requested && !validation_enabled {
            tracing::warn!(
                "Vulkan API validation requested but {} is not installed; install the Khronos \
                 validation layers (e.g. the `vulkan-validation-layers` package) to enable it",
                VALIDATION_LAYER_NAME.to_string_lossy()
            );
        }

        let layers: Vec<*const std::ffi::c_char> = if validation_enabled {
            vec![VALIDATION_LAYER_NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let mut extensions: Vec<*const std::ffi::c_char> = if debug_utils_enabled {
            vec![ash::ext::debug_utils::NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let surface_extensions = surface_extensions(&entry);
        extensions.extend(surface_extensions.iter().map(|name| name.as_ptr()));
        // Portability implementations (MoltenVK on macOS) are only enumerated with this
        // extension and flag (RVK-14).
        let portability =
            instance_extension_offered(&entry, ash::khr::portability_enumeration::NAME);
        if portability {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }

        // Synchronization validation (missing or wrong barriers) through VK_EXT_layer_settings,
        // on whenever the validation layer is, unless KORPPI_RHI_SYNC_VALIDATION=0.
        let sync_validation = validation_enabled
            && sync_validation_requested()
            && layer_extension_offered(&entry, VALIDATION_LAYER_NAME, LAYER_SETTINGS_EXTENSION);
        if sync_validation {
            extensions.push(LAYER_SETTINGS_EXTENSION.as_ptr());
        }
        let validate_sync: vk::Bool32 = vk::TRUE;
        let layer_settings = [vk::LayerSettingEXT {
            p_layer_name: VALIDATION_LAYER_NAME.as_ptr(),
            p_setting_name: c"validate_sync".as_ptr(),
            ty: vk::LayerSettingTypeEXT::BOOL32,
            value_count: 1,
            p_values: (&raw const validate_sync).cast(),
            ..Default::default()
        }];
        let mut layer_settings_info =
            vk::LayerSettingsCreateInfoEXT::default().settings(&layer_settings);

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"KorppiApp")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"Korppi")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(if portability {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::empty()
            });
        let instance_create_info = if sync_validation {
            instance_create_info.push_next(&mut layer_settings_info)
        } else {
            instance_create_info
        };

        // SAFETY: `instance_create_info` and every string it points to outlive this call.
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(|e| {
                    RhiError::InitializationFailed(format!("Failed to create Vulkan instance: {e}"))
                })?
        };

        let router = Box::new(DiagnosticRouter::default());
        let debug_messenger = if debug_utils_enabled && validation_requested {
            create_debug_messenger(&entry, &instance, &router)
        } else {
            None
        };
        let surface_loader = (!surface_extensions.is_empty())
            .then(|| ash::khr::surface::Instance::new(&entry, &instance));
        Ok(Self {
            entry,
            instance,
            surface_extensions,
            surface_loader,
            validation_enabled,
            debug_utils_enabled,
            router,
            debug_messenger,
        })
    }

    /// The router of validation messages, if a `VK_EXT_debug_utils` messenger is installed.
    pub(crate) fn diagnostic_router(&self) -> Option<&DiagnosticRouter> {
        self.debug_messenger.as_ref().map(|_| &*self.router)
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        // SAFETY: the last `SharedInstance` is gone, so every device created from the instance
        // has been destroyed. The messenger is destroyed before the instance it belongs to, and
        // before the router its user data points to.
        unsafe {
            if let Some(m) = &self.debug_messenger {
                m.loader.destroy_debug_utils_messenger(m.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// A device's hold on the process-wide [`VulkanInstance`].
///
/// Acquiring the first hold creates the instance; releasing the last one destroys it. Both
/// happen under one lock, so a new instance is never created while the previous one is being
/// destroyed.
pub(crate) struct SharedInstance(ManuallyDrop<Arc<VulkanInstance>>);

impl SharedInstance {
    /// Returns a hold on the process's instance, creating it if no device holds one.
    pub(crate) fn acquire() -> Result<Self, RhiError> {
        let mut shared = lock(&SHARED);
        let instance = match &*shared {
            Some(instance) => instance.clone(),
            None => {
                let instance = Arc::new(VulkanInstance::create()?);
                *shared = Some(instance.clone());
                instance
            }
        };
        Ok(Self(ManuallyDrop::new(instance)))
    }
}

impl Deref for SharedInstance {
    type Target = VulkanInstance;

    fn deref(&self) -> &VulkanInstance {
        &self.0
    }
}

impl Drop for SharedInstance {
    fn drop(&mut self) {
        let mut shared = lock(&SHARED);
        // SAFETY: `self.0` is taken once, here, and not used afterwards.
        drop(unsafe { ManuallyDrop::take(&mut self.0) });
        // Clones of the `Arc` exist only in `SHARED` and in holds, so a count of one means that
        // no device holds the instance any more: destroy it before releasing the lock.
        if shared
            .as_ref()
            .is_some_and(|instance| Arc::strong_count(instance) == 1)
        {
            *shared = None;
        }
    }
}

/// Sends each validation message to the [`DiagnosticLog`] of the device it concerns.
///
/// Devices register their `VkDevice`, queues and command buffers, the dispatchable handles that
/// validation messages name for device-level calls; those handles are unique among live objects
/// in the process. A message that names none of them (instance-level calls, the loader) concerns
/// every device of the instance and reaches all of them.
#[derive(Default)]
pub(crate) struct DiagnosticRouter {
    routes: Mutex<Routes>,
}

#[derive(Default)]
struct Routes {
    /// The log of each registered dispatchable handle.
    objects: HashMap<u64, Arc<DiagnosticLog>>,
    /// The log of every registered device.
    devices: Vec<Arc<DiagnosticLog>>,
}

impl DiagnosticRouter {
    /// Routes messages about `device` (and messages that name no device) to `log`.
    pub(crate) fn register_device(&self, device: vk::Device, log: &Arc<DiagnosticLog>) {
        let mut routes = lock(&self.routes);
        routes.objects.insert(device.as_raw(), log.clone());
        routes.devices.push(log.clone());
    }

    /// Stops routing to `log`. Call it after the device is destroyed, so that messages about
    /// its destruction still reach it; a device created since with the same handle keeps its
    /// own route.
    pub(crate) fn unregister_device(&self, device: vk::Device, log: &Arc<DiagnosticLog>) {
        let mut routes = lock(&self.routes);
        remove_route(&mut routes.objects, device.as_raw(), log);
        routes.devices.retain(|l| !Arc::ptr_eq(l, log));
    }

    /// Routes messages about `object`, a queue or command buffer, to `log`.
    pub(crate) fn register_object(&self, object: impl Handle, log: &Arc<DiagnosticLog>) {
        lock(&self.routes)
            .objects
            .insert(object.as_raw(), log.clone());
    }

    /// Stops routing messages about `object` to `log`; call it after the object is destroyed.
    pub(crate) fn unregister_object(&self, object: impl Handle, log: &Arc<DiagnosticLog>) {
        remove_route(&mut lock(&self.routes).objects, object.as_raw(), log);
    }

    /// Appends a message to the log of the first registered device, queue or command buffer
    /// among `objects`, or to every device's log if it names none.
    fn route(
        &self,
        objects: &[vk::DebugUtilsObjectNameInfoEXT<'_>],
        severity: DiagnosticSeverity,
        message: &str,
    ) {
        let routes = lock(&self.routes);
        let owner = objects
            .iter()
            .filter(|o| {
                matches!(
                    o.object_type,
                    vk::ObjectType::DEVICE | vk::ObjectType::QUEUE | vk::ObjectType::COMMAND_BUFFER
                )
            })
            .find_map(|o| routes.objects.get(&o.object_handle));
        match owner {
            Some(log) => log.push(severity, message),
            None => {
                for log in &routes.devices {
                    log.push(severity, message);
                }
            }
        }
    }
}

/// Removes the route of `handle` if it leads to `log`.
fn remove_route(
    objects: &mut HashMap<u64, Arc<DiagnosticLog>>,
    handle: u64,
    log: &Arc<DiagnosticLog>,
) {
    if objects.get(&handle).is_some_and(|l| Arc::ptr_eq(l, log)) {
        objects.remove(&handle);
    }
}

fn sync_validation_requested() -> bool {
    std::env::var(SYNC_VALIDATION_ENV_VAR).map_or(true, |v| v != "0")
}

/// True if `layer` offers instance extension `name`.
fn layer_extension_offered(entry: &Entry, layer: &CStr, name: &CStr) -> bool {
    // SAFETY: enumerating a layer's instance extensions has no preconditions beyond a loaded
    // `Entry`; `layer` is NUL-terminated.
    unsafe { entry.enumerate_instance_extension_properties(Some(layer)) }
        .unwrap_or_default()
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(name))
}

/// `VK_KHR_surface` plus every platform surface extension the loader offers, or nothing if it
/// does not offer `VK_KHR_surface`.
fn surface_extensions(entry: &Entry) -> Vec<&'static CStr> {
    // SAFETY: enumerating instance extensions has no preconditions beyond a loaded `Entry`.
    let available =
        unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
    let offered = |name: &CStr| {
        available
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(name))
    };
    if !offered(ash::khr::surface::NAME) {
        return Vec::new();
    }
    std::iter::once(ash::khr::surface::NAME)
        .chain(
            PLATFORM_SURFACE_EXTENSIONS
                .into_iter()
                .filter(|name| offered(name)),
        )
        .collect()
}

/// True if the loader offers instance extension `name`.
fn instance_extension_offered(entry: &Entry, name: &CStr) -> bool {
    // SAFETY: enumerating instance extensions has no preconditions beyond a loaded `Entry`.
    unsafe { entry.enumerate_instance_extension_properties(None) }
        .unwrap_or_default()
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(name))
}

/// True if the Khronos validation layer is installed.
fn layer_available(entry: &Entry) -> bool {
    // SAFETY: enumerating instance layers has no preconditions beyond a loaded `Entry`.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
    layers
        .iter()
        .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER_NAME))
}

/// True if `VK_EXT_debug_utils` is offered by the loader/ICDs (`layer == None`) or by `layer`.
fn debug_utils_available(entry: &Entry, layer: Option<&CStr>) -> bool {
    // SAFETY: enumerating instance extensions has no preconditions beyond a loaded `Entry`; the
    // optional layer name is a valid NUL-terminated string.
    let extensions =
        unsafe { entry.enumerate_instance_extension_properties(layer) }.unwrap_or_default();
    extensions
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME))
}

fn create_debug_messenger(
    entry: &Entry,
    instance: &Instance,
    router: &DiagnosticRouter,
) -> Option<DebugMessenger> {
    let loader = ash::ext::debug_utils::Instance::new(entry, instance);
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_messenger_callback))
        .user_data(router as *const DiagnosticRouter as *mut c_void);
    // SAFETY: `VK_EXT_debug_utils` is enabled on `instance`. The user-data pointer targets the
    // boxed `DiagnosticRouter` of the `VulkanInstance` that will own the messenger, which
    // destroys the messenger before the router (`VulkanInstance::drop`).
    match unsafe { loader.create_debug_utils_messenger(&create_info, None) } {
        Ok(messenger) => Some(DebugMessenger { loader, messenger }),
        Err(error) => {
            tracing::warn!("Failed to create Vulkan debug messenger: {error}");
            None
        }
    }
}

/// Maps a `VK_EXT_debug_utils` severity to the RHI's diagnostic severity.
fn diagnostic_severity(flags: vk::DebugUtilsMessageSeverityFlagsEXT) -> DiagnosticSeverity {
    if flags.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        DiagnosticSeverity::Error
    } else if flags.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        DiagnosticSeverity::Warning
    } else {
        DiagnosticSeverity::Info
    }
}

/// `VK_EXT_debug_utils` messenger callback: logs through `tracing` and routes the message with
/// the [`DiagnosticRouter`] passed as user data. Never panics and always returns `VK_FALSE`.
unsafe extern "system" fn debug_messenger_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    p_user_data: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the implementation passes a valid callback-data struct, or null, for the duration
    // of the call.
    let data = unsafe { p_callback_data.as_ref() };
    let message = data
        .and_then(|data| {
            // SAFETY: as above; `message_as_c_str` handles a null `pMessage`.
            unsafe { data.message_as_c_str() }.map(|m| m.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| String::from("<no callback data>"));
    let objects = match data {
        Some(data) if data.object_count > 0 && !data.p_objects.is_null() => {
            // SAFETY: `pObjects` points to `objectCount` valid structs for the duration of the
            // call.
            unsafe { std::slice::from_raw_parts(data.p_objects, data.object_count as usize) }
        }
        _ => &[],
    };
    let severity = diagnostic_severity(message_severity);
    match severity {
        DiagnosticSeverity::Error => {
            tracing::error!(target: "korppi_rhi_vulkan::validation", "{message}")
        }
        DiagnosticSeverity::Warning => {
            tracing::warn!(target: "korppi_rhi_vulkan::validation", "{message}")
        }
        DiagnosticSeverity::Info => {
            tracing::debug!(target: "korppi_rhi_vulkan::validation", "{message}")
        }
    }
    // SAFETY: the messenger was created with a pointer to a `DiagnosticRouter` that outlives it
    // (see `create_debug_messenger`), or the pointer is null; the router is `Sync`, so shared
    // access from any calling thread is sound.
    if let Some(router) = unsafe { (p_user_data as *const DiagnosticRouter).as_ref() } {
        router.route(objects, severity, &message);
    }
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(ty: vk::ObjectType, handle: u64) -> vk::DebugUtilsObjectNameInfoEXT<'static> {
        vk::DebugUtilsObjectNameInfoEXT {
            object_type: ty,
            object_handle: handle,
            ..Default::default()
        }
    }

    /// Delivers a message naming `objects` through the messenger callback.
    fn deliver(router: &DiagnosticRouter, objects: &[vk::DebugUtilsObjectNameInfoEXT<'_>]) {
        let data = vk::DebugUtilsMessengerCallbackDataEXT::default()
            .message(c"VUID-test-00000: invalid usage")
            .objects(objects);
        // SAFETY: `data`, `objects` and `router` outlive the call, as the messenger guarantees.
        let result = unsafe {
            debug_messenger_callback(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                &data,
                router as *const DiagnosticRouter as *mut c_void,
            )
        };
        assert_eq!(result, vk::FALSE);
    }

    #[test]
    fn severity_mapping_prefers_the_most_severe_bit() {
        use vk::DebugUtilsMessageSeverityFlagsEXT as S;
        assert_eq!(diagnostic_severity(S::ERROR), DiagnosticSeverity::Error);
        assert_eq!(
            diagnostic_severity(S::WARNING | S::ERROR),
            DiagnosticSeverity::Error
        );
        assert_eq!(diagnostic_severity(S::WARNING), DiagnosticSeverity::Warning);
        assert_eq!(diagnostic_severity(S::INFO), DiagnosticSeverity::Info);
        assert_eq!(diagnostic_severity(S::VERBOSE), DiagnosticSeverity::Info);
    }

    #[test]
    fn messages_reach_the_device_of_the_object_they_name() {
        let router = DiagnosticRouter::default();
        let (first, second) = (
            Arc::new(DiagnosticLog::new()),
            Arc::new(DiagnosticLog::new()),
        );
        router.register_device(vk::Device::from_raw(0x10), &first);
        router.register_device(vk::Device::from_raw(0x20), &second);
        router.register_object(vk::CommandBuffer::from_raw(0x30), &second);

        deliver(&router, &[object(vk::ObjectType::DEVICE, 0x10)]);
        let entries = first.take();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].severity, DiagnosticSeverity::Error);
        assert_eq!(entries[0].message, "VUID-test-00000: invalid usage");
        assert!(second.take().is_empty());

        // The first dispatchable object a device registered decides; other objects are skipped.
        deliver(
            &router,
            &[
                object(vk::ObjectType::IMAGE, 0x10),
                object(vk::ObjectType::COMMAND_BUFFER, 0x30),
            ],
        );
        assert!(first.take().is_empty());
        assert_eq!(second.take().len(), 1);
    }

    #[test]
    fn messages_that_name_no_device_reach_every_device() {
        let router = DiagnosticRouter::default();
        let (first, second) = (
            Arc::new(DiagnosticLog::new()),
            Arc::new(DiagnosticLog::new()),
        );
        router.register_device(vk::Device::from_raw(0x10), &first);
        router.register_device(vk::Device::from_raw(0x20), &second);
        deliver(&router, &[object(vk::ObjectType::INSTANCE, 0x99)]);
        deliver(&router, &[]);
        assert_eq!(first.take().len(), 2);
        assert_eq!(second.take().len(), 2);
    }

    #[test]
    fn unregistering_keeps_a_newer_route_for_the_same_handle() {
        let router = DiagnosticRouter::default();
        let (old, new) = (
            Arc::new(DiagnosticLog::new()),
            Arc::new(DiagnosticLog::new()),
        );
        let device = vk::Device::from_raw(0x10);
        router.register_device(device, &old);
        // The handle is reused by a device created before the old one unregistered.
        router.register_device(device, &new);
        router.unregister_device(device, &old);
        deliver(&router, &[object(vk::ObjectType::DEVICE, 0x10)]);
        assert!(old.take().is_empty());
        assert_eq!(new.take().len(), 1);
    }

    #[test]
    fn callback_tolerates_null_pointers() {
        // SAFETY: the callback documents that null callback data and user data are accepted.
        let result = unsafe {
            debug_messenger_callback(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, vk::FALSE);
    }
}
