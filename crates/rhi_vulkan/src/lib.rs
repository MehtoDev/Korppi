//! Vulkan 1.3 Core Render Hardware Interface backend (`korppi_rhi_vulkan`).
//!
//! Provides a low-overhead, explicit Vulkan 1.3 implementation utilizing:
//! - `VK_KHR_dynamic_rendering` for lightweight pass management without `VkRenderPass`/`VkFramebuffer`.
//! - `VK_KHR_synchronization2` for 64-bit pipeline stages and explicit memory barriers.
//! - `VK_KHR_timeline_semaphore` for monotonic queue synchronization.
//! - Suballocated chunk memory pools to respect the hardware `maxMemoryAllocationCount` limit.
//! - Negative viewport height for unified $+Y$-up coordinate space.

/// Bind group layouts and bind groups (descriptor sets).
pub mod binding;
/// Adapter information, limits, features and format support.
pub mod capabilities;
/// Command buffer recording and execution.
pub mod command;
/// Physical and logical device context on the process-wide instance.
pub mod context;
/// Device implementation of GraphicsDevice.
pub mod device;
/// Mapping of Vulkan results to `RhiError`.
mod error;
/// The process-wide Vulkan instance and the routing of its validation messages to devices.
pub mod instance;
/// Chunk-based GPU memory pool suballocator.
pub mod memory;
/// Pipelines with shared layouts, shared PSOs and extended dynamic state.
mod pipeline;
/// The device's pipeline cache and its persistence.
mod pipeline_cache;
/// Timestamp query pools and debug names.
pub mod query;
/// Graphics-queue submission tracked by a timeline semaphore.
mod queue;
/// Per-recorder command pools, recycled after completion.
mod recording;
/// Internal Vulkan resource representations.
pub mod resource;
/// Swapchain presentation.
pub mod swapchain;
/// Textures and texture views.
mod texture;
/// Type mappings and format conversions between RHI and Vulkan.
pub mod types;

pub use command::{VulkanCommandBuffer, VulkanCommandBufferRecorder};
pub use context::VulkanContext;
pub use device::VulkanDevice;
pub use memory::{HeapBudget, Suballocation, VulkanMemoryManager, DEFAULT_CHUNK_SIZE};
pub use recording::RecordingStats;
