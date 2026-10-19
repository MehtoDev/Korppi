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
