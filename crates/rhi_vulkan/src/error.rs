//! Mapping of Vulkan results to `RhiError` (finding RVK-18).

use ash::vk;
use korppi_rhi::RhiError;

/// The `RhiError` for `result` returned by Vulkan command `call`: device loss and memory
/// exhaustion keep their meaning wherever they occur, so callers can react to them (for
/// example by recreating the device after `DeviceLost`).
pub(crate) fn vk_error(call: &str, result: vk::Result) -> RhiError {
    match result {
        vk::Result::ERROR_DEVICE_LOST => {
            tracing::error!("{call} reported VK_ERROR_DEVICE_LOST");
            RhiError::DeviceLost
        }
        vk::Result::ERROR_OUT_OF_HOST_MEMORY
        | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        | vk::Result::ERROR_OUT_OF_POOL_MEMORY
        | vk::Result::ERROR_FRAGMENTED_POOL => {
            RhiError::OutOfMemory(format!("{call} failed: {result}"))
        }
        _ => RhiError::BackendError(format!("{call} failed: {result}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_loss_and_memory_exhaustion_keep_their_meaning() {
        assert_eq!(
            vk_error("vkQueueSubmit2", vk::Result::ERROR_DEVICE_LOST),
            RhiError::DeviceLost
        );
        for result in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ] {
            assert!(matches!(
                vk_error("vkAllocateMemory", result),
                RhiError::OutOfMemory(message) if message.contains("vkAllocateMemory")
            ));
        }
        assert!(matches!(
            vk_error("vkWaitSemaphores", vk::Result::ERROR_UNKNOWN),
            RhiError::BackendError(_)
        ));
    }
}
