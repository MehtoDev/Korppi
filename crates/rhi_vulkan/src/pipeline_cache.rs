//! The device's `VkPipelineCache` and its persistence (finding RVK-16).
//!
//! Every pipeline is created through one cache. `GraphicsDevice::pipeline_cache_data` returns
//! its data behind a Korppi tag; `load_pipeline_cache` merges data of an earlier run after
//! checking the Vulkan cache header (vendor, device, cache UUID), so data of another driver or
//! GPU is ignored rather than handed to a driver that might not validate it. Data with a
//! matching header that the driver still refuses (a truncated or damaged file) is ignored too.

use crate::context::VulkanContext;
use ash::vk;
use korppi_rhi::RhiError;
use std::sync::{Arc, RwLock, RwLockReadGuard};

/// Tag in front of serialized cache data.
const MAGIC: &[u8] = b"KORPPI-VK-PIPELINE-CACHE\0";
/// Version of the tagged format.
const VERSION: u32 = 1;
/// Size of `VkPipelineCacheHeaderVersionOne`.
const VK_HEADER_SIZE: usize = 32;

/// The device's pipeline cache. Pipeline creation reads it concurrently; merging takes it
/// exclusively (`vkMergePipelineCaches` needs the destination externally synchronized).
pub(crate) struct PipelineCache {
    context: Arc<VulkanContext>,
    cache: RwLock<vk::PipelineCache>,
}

impl PipelineCache {
    pub(crate) fn new(context: Arc<VulkanContext>) -> Result<Self, RhiError> {
        // SAFETY: the create info is valid; the cache is destroyed on drop.
        let cache = unsafe {
            context
                .device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        }
        .map_err(|e| RhiError::OutOfMemory(format!("vkCreatePipelineCache failed: {e}")))?;
        Ok(Self {
            context,
            cache: RwLock::new(cache),
        })
    }

    /// The cache, for pipeline creation.
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, vk::PipelineCache> {
        self.cache
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The cache's data behind the Korppi tag.
    pub(crate) fn data(&self) -> Result<Vec<u8>, RhiError> {
        let cache = self.read();
        // SAFETY: the cache is live; creation may use it concurrently (caches are internally
        // synchronized for that).
        let data = unsafe { self.context.device.get_pipeline_cache_data(*cache) }
            .map_err(|e| crate::error::vk_error("vkGetPipelineCacheData", e))?;
        // Data without a valid header for this device (lavapipe returns a 20-byte stub) cannot
        // be loaded again: the device keeps no cache worth persisting.
        if !self.header_matches(&data) {
            return Ok(Vec::new());
        }
        let mut tagged = Vec::with_capacity(MAGIC.len() + 4 + data.len());
        tagged.extend_from_slice(MAGIC);
        tagged.extend_from_slice(&VERSION.to_le_bytes());
        tagged.extend_from_slice(&data);
        Ok(tagged)
    }

    /// Merges tagged `data` if it belongs to this device; returns whether it did.
    pub(crate) fn load(&self, data: &[u8]) -> Result<bool, RhiError> {
        let Some(vk_data) = data
            .strip_prefix(MAGIC)
            .and_then(|rest| rest.strip_prefix(&VERSION.to_le_bytes()[..]))
        else {
            return Ok(false);
        };
        if !self.header_matches(vk_data) {
            return Ok(false);
        }
        let info = vk::PipelineCacheCreateInfo::default().initial_data(vk_data);
        // SAFETY: the header was checked against this device; the source cache is destroyed
        // below.
        let source = match unsafe { self.context.device.create_pipeline_cache(&info, None) } {
            Ok(source) => source,
            Err(error) => return refused_data(error),
        };
        let merged = {
            let cache = self
                .cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // SAFETY: the write lock gives exclusive access to the destination cache.
            unsafe {
                self.context
                    .device
                    .merge_pipeline_caches(*cache, std::slice::from_ref(&source))
            }
        };
        // SAFETY: the source cache is no longer used.
        unsafe { self.context.device.destroy_pipeline_cache(source, None) };
        merged
            .map(|()| true)
            .map_err(|e| crate::error::vk_error("vkMergePipelineCaches", e))
    }

    /// True if `vk_data` starts with a version-one header of this physical device.
    fn header_matches(&self, vk_data: &[u8]) -> bool {
        if vk_data.len() < VK_HEADER_SIZE {
            return false;
        }
        let word = |i: usize| u32::from_le_bytes(vk_data[i * 4..i * 4 + 4].try_into().unwrap());
        // SAFETY: the physical device belongs to the instance.
        let properties = unsafe {
            self.context
                .instance
                .get_physical_device_properties(self.context.physical_device)
        };
        word(0) as usize >= VK_HEADER_SIZE
            && word(1) == vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32
            && word(2) == properties.vendor_id
            && word(3) == properties.device_id
            && vk_data[16..32] == properties.pipeline_cache_uuid
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        let cache = *self.read();
        // SAFETY: the device drops this after waiting for idle; no pipeline creation is running.
        unsafe { self.context.device.destroy_pipeline_cache(cache, None) };
    }
}

/// The result of loading data that `vkCreatePipelineCache` refused with `error`. A matching
/// header does not make the rest usable (a truncated or damaged file; NVIDIA reports
/// `VK_ERROR_UNKNOWN`, ISSUE-009), so such data is ignored like foreign data and the cache stays
/// as it was; running out of memory is still an error.
fn refused_data(error: vk::Result) -> Result<bool, RhiError> {
    match error {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            Err(crate::error::vk_error("vkCreatePipelineCache", error))
        }
        _ => {
            tracing::warn!("pipeline cache data refused by the driver ({error}); not loaded");
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ISSUE-009: data the driver refuses (NVIDIA returns `VK_ERROR_UNKNOWN` for truncated data
    /// behind a matching header) is ignored like foreign data; running out of memory is not.
    #[test]
    fn data_the_driver_refuses_is_ignored_but_out_of_memory_is_reported() {
        for refused in [
            vk::Result::ERROR_UNKNOWN,
            vk::Result::ERROR_INITIALIZATION_FAILED,
            vk::Result::ERROR_VALIDATION_FAILED_EXT,
        ] {
            assert!(matches!(refused_data(refused), Ok(false)), "{refused}");
        }
        for out_of_memory in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ] {
            assert!(
                matches!(refused_data(out_of_memory), Err(RhiError::OutOfMemory(_))),
                "{out_of_memory}"
            );
        }
    }
}
