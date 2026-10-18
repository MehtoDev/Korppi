//! Timestamp query pools and debug names (FIX-1.4).

use crate::device::{lock, Garbage, VulkanDevice};
use ash::vk;
use korppi_rhi::{
    DeviceFeatures, GraphicsDevice, QuerySetDescriptor, QuerySetHandle, QueryType, RhiError,
};
use std::ffi::CString;

/// Internal Vulkan query set: a query pool.
pub struct VulkanQuerySet {
    /// The query pool.
    pub pool: vk::QueryPool,
    /// Number of queries.
    pub count: u32,
}

impl VulkanDevice {
    /// Names `handle` for debuggers and the validation layer (`VK_EXT_debug_utils`), if the
    /// extension is enabled and `label` is given.
    pub(crate) fn set_object_name<H: vk::Handle>(&self, handle: H, label: Option<&str>) {
        let (Some(debug_utils), Some(label)) = (&self.context.debug_utils, label) else {
            return;
        };
        let Ok(name) = CString::new(label) else {
            return;
        };
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&name);
        // SAFETY: `handle` is a live object of this device; `info` outlives the call.
        if let Err(error) = unsafe { debug_utils.set_debug_utils_object_name(&info) } {
            tracing::debug!("vkSetDebugUtilsObjectNameEXT failed: {error}");
        }
    }

    pub(crate) fn create_query_set_impl(
        &self,
        desc: &QuerySetDescriptor,
    ) -> Result<QuerySetHandle, RhiError> {
        match desc.ty {
            QueryType::Timestamp => {
                if !self.features().contains(DeviceFeatures::TIMESTAMP_QUERY) {
                    return Err(RhiError::Unsupported(
                        "the graphics queue has no valid timestamp bits".into(),
                    ));
                }
            }
        }
        if desc.count == 0 {
            return Err(RhiError::ValidationError(
                "query sets need at least one query".into(),
            ));
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(desc.count);
        // SAFETY: `info` is valid; the pool is destroyed through deferred destruction.
        let pool = unsafe { self.context.device.create_query_pool(&info, None) }
            .map_err(|e| RhiError::OutOfMemory(format!("vkCreateQueryPool failed: {e}")))?;
        // Queries start unavailable only after a reset; resolves reset them again.
        let reset = self.submit_one_time(|device, cmd| {
            // SAFETY: `cmd` is recording outside a render pass; the range is the whole pool.
            unsafe { device.cmd_reset_query_pool(cmd, pool, 0, desc.count) };
        });
        if let Err(error) = reset {
            // SAFETY: the pool is unused.
            unsafe { self.context.device.destroy_query_pool(pool, None) };
            return Err(error);
        }
        self.set_object_name(pool, desc.label);
        let (id, generation) = lock(&self.resources.query_sets).insert(VulkanQuerySet {
            pool,
            count: desc.count,
        });
        Ok(QuerySetHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    pub(crate) fn destroy_query_set_impl(&self, handle: QuerySetHandle) -> Result<(), RhiError> {
        let (set, _) = lock(&self.resources.query_sets)
            .remove(handle)
            .ok_or_else(|| RhiError::InvalidHandle(format!("query set {handle:?} not found")))?;
        self.defer_destruction(Garbage::QueryPool(set.pool))
    }
}
