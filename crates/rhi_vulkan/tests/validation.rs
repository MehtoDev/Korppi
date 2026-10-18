//! Integration tests for API-level validation capture (`VK_EXT_debug_utils`, Khronos validation
//! layer) on the headless Vulkan device.
//!
//! Tests that need the validation layer print a notice and return early when it is not
//! installed (install the `vulkan-validation-layers` package to run them fully).

use ash::vk;
use korppi_rhi::{api_validation_requested, DiagnosticSeverity, GraphicsDevice};
use korppi_rhi_vulkan::context::VALIDATION_LAYER_NAME;
use korppi_rhi_vulkan::VulkanDevice;

fn validation_layer_installed(entry: &ash::Entry) -> bool {
    // SAFETY: enumerating instance layers has no preconditions beyond a loaded `Entry`.
    unsafe { entry.enumerate_instance_layer_properties() }
        .unwrap_or_default()
        .iter()
        .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER_NAME))
}

#[test]
fn validation_layer_is_enabled_exactly_when_requested_and_installed() {
    let device = VulkanDevice::create_headless().expect("Vulkan device");
    let expected =
        api_validation_requested() && validation_layer_installed(&device.context().entry);
    assert_eq!(device.api_validation_enabled(), expected);
}

#[test]
fn debug_utils_messages_reach_take_diagnostics() {
    let device = VulkanDevice::create_headless().expect("Vulkan device");
    let context = device.context();
    if !context.debug_messenger_installed() {
        assert!(
            !api_validation_requested(),
            "validation was requested, but no VK_EXT_debug_utils messenger was installed"
        );
        eprintln!("skipped: API validation is disabled for this process");
        return;
    }
    let _ = device.take_diagnostics();
    let loader = ash::ext::debug_utils::Instance::new(&context.entry, &context.instance);
    let data =
        vk::DebugUtilsMessengerCallbackDataEXT::default().message(c"korppi debug-utils round trip");
    // SAFETY: `VK_EXT_debug_utils` is enabled on this instance (a messenger is installed) and
    // `data` outlives the call.
    unsafe {
        loader.submit_debug_utils_message(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            &data,
        );
    }
    let diagnostics = device.take_diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error
                && d.message.contains("korppi debug-utils round trip")),
        "submitted message not collected: {diagnostics:?}"
    );
}

#[test]
fn validation_layer_reports_invalid_usage() {
    let device = VulkanDevice::create_headless().expect("Vulkan device");
    if !device.api_validation_enabled() {
        eprintln!(
            "skipped: {} is not enabled (not installed, or validation disabled)",
            VALIDATION_LAYER_NAME.to_string_lossy()
        );
        return;
    }
    let _ = device.take_diagnostics();
    // VUID-VkBufferCreateInfo-size-00912: size must be greater than 0.
    let info = vk::BufferCreateInfo::default()
        .size(0)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
    let raw = &device.context().device;
    // SAFETY: deliberately invalid usage that the validation layer intercepts and reports; the
    // buffer, if the driver creates one anyway, is destroyed immediately and never used.
    unsafe {
        if let Ok(buffer) = raw.create_buffer(&info, None) {
            raw.destroy_buffer(buffer, None);
        }
    }
    let diagnostics = device.take_diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error
                && d.message.contains("VUID-VkBufferCreateInfo-size-00912")),
        "expected VUID-VkBufferCreateInfo-size-00912, got {diagnostics:?}"
    );
}
