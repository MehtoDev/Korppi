//! Foundation utilities, timing, memory management, and diagnostics for the engine.

pub mod arena;
pub mod bitflags;
pub mod clock;
pub mod id_pool;
pub mod log;
pub mod pod;
pub mod slot_map;
pub mod smallvec;
pub mod task;

pub use arena::FrameArena;
pub use clock::FixedClock;
pub use id_pool::{IdPool, IdPoolParts};
pub use log::init_logging;
pub use pod::{
    bytes_of, bytes_of_mut, cast_slice, cast_slice_mut, pod_read_unaligned, Pod, Zeroable,
};
pub use slot_map::{Handle, SlotMap};
pub use smallvec::{Array as SmallVecArray, IntoIter as SmallVecIntoIter, SmallVec};
pub use task::{parallel_for, spin_until, SerialExecutor, TaskExecutor, WorkerPool};

/// Adds two unsigned 64-bit integers together.
///
/// # Examples
/// ```
/// use korppi_core::add;
/// assert_eq!(add(2, 3), 5);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
