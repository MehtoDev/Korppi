//! Generational resource handles (finding RHI-10) that carry the device that created them
//! (finding RHI-15).

use std::sync::atomic::{AtomicU32, Ordering};

/// The identity of a device. Every handle carries the identity of the device that created it,
/// and a device rejects the handles of other devices: each device numbers its resources from
/// slot 0, so without it another device's handle named a resource of its own (RHI-15).
///
/// Identities come from one process-wide counter, so no two devices share one, even after one
/// of them is dropped (the counter would have to pass 2^32 - 1 devices to repeat). [`DeviceId::NONE`]
/// marks handles that no device created: the null sentinel and handles built with `new`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceId(u32);

impl DeviceId {
    /// No device.
    pub const NONE: Self = Self(0);

    /// A new identity, distinct from every other one in the process.
    pub fn next() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(1);
        loop {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            if id != 0 {
                return Self(id);
            }
        }
    }

    /// The identity as a number (0 for [`DeviceId::NONE`]).
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Macro to define a strongly-typed generational resource handle.
macro_rules! define_handle {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $name {
            id: u32,
            generation: u32,
            device: DeviceId,
        }

        impl $name {
            /// Creates a handle with the specified slot id and generation counter and no
            /// device ([`DeviceId::NONE`]); devices create theirs with [`Self::from_parts`].
            #[inline]
            pub const fn new(id: u32, generation: u32) -> Self {
                Self {
                    id,
                    generation,
                    device: DeviceId::NONE,
                }
            }

            /// Creates a handle of `device`'s slot `id` at `generation`.
            #[inline]
            pub const fn from_parts(device: DeviceId, id: u32, generation: u32) -> Self {
                Self {
                    id,
                    generation,
                    device,
                }
            }

            /// Creates a null/invalid handle sentinel.
            #[inline]
            pub const fn null() -> Self {
                Self {
                    id: u32::MAX,
                    generation: 0,
                    device: DeviceId::NONE,
                }
            }

            /// Returns the slot index of the resource.
            #[inline]
            pub const fn id(&self) -> u32 {
                self.id
            }

            /// Returns the generation count of the resource.
            #[inline]
            pub const fn generation(&self) -> u32 {
                self.generation
            }

            /// Returns the device that created the resource.
            #[inline]
            pub const fn device(&self) -> DeviceId {
                self.device
            }

            /// Returns true if this handle is not the null sentinel.
            #[inline]
            pub const fn is_valid(&self) -> bool {
                self.id != u32::MAX
            }
        }

        impl Default for $name {
            #[inline]
            fn default() -> Self {
                Self::null()
            }
        }

        impl crate::slot_map::SlotKey for $name {
            #[inline]
            fn slot(&self) -> u32 {
                self.id
            }

            #[inline]
            fn slot_generation(&self) -> u32 {
                self.generation
            }

            #[inline]
            fn slot_device(&self) -> DeviceId {
                self.device
            }
        }

        impl crate::slot_map::SlotHandle for $name {
            #[inline]
            fn from_slot(device: DeviceId, id: u32, generation: u32) -> Self {
                Self::from_parts(device, id, generation)
            }
        }
    };
}

define_handle!(
    BufferHandle,
    "Generational handle referencing a GPU buffer."
);
define_handle!(
    TextureHandle,
    "Generational handle referencing a GPU texture."
);
define_handle!(
    TextureViewHandle,
    "Generational handle referencing a view of a range of a texture's mips, layers and aspects."
);
define_handle!(
    SamplerHandle,
    "Generational handle referencing a texture sampler."
);
define_handle!(
    ShaderHandle,
    "Generational handle referencing a compiled shader stage."
);
define_handle!(
    PipelineHandle,
    "Generational handle referencing a graphics or compute pipeline."
);
define_handle!(
    BindGroupLayoutHandle,
    "Generational handle referencing a bind group (descriptor set) layout."
);
define_handle!(
    BindGroupHandle,
    "Generational handle referencing a bind group: a layout filled with resources."
);
define_handle!(
    QuerySetHandle,
    "Generational handle referencing a set of GPU queries."
);
define_handle!(
    SwapchainHandle,
    "Generational handle referencing a presentation swapchain."
);
