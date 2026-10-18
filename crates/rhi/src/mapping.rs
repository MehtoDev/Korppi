//! Host access to buffer memory (finding RHI-05): `GraphicsDevice::map_buffer`.
//!
//! `CpuToGpu` and `GpuToCpu` buffers stay mapped for their whole lifetime (persistent mapping);
//! `map_buffer` hands out a [`MappedBuffer`] guard over that memory. The guard copies data in
//! and out (`write`, `read`) and exposes the pointer for code that streams into it directly.
//!
//! Safety model:
//! - Host access is exclusive per buffer: while a guard exists, mapping the buffer again,
//!   `write_buffer`, `read_buffer` and `destroy_buffer` fail with `ValidationError`, so CPU
//!   threads never race on the memory and it is never freed under the guard.
//! - The guard borrows the device, which therefore outlives it.
//! - No Rust reference into mapped memory is ever created; the GPU may access the memory while
//!   it is mapped. Ordering host access against GPU work is the caller's job, as for
//!   `write_buffer`: write only ranges no pending submission reads (one region per frame in
//!   flight, `FRAMES_IN_FLIGHT`), and read results after the submission that wrote them has
//!   completed and recorded a barrier to `AccessFlags::HOST_READ`.

use crate::error::RhiError;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Per-buffer lock that makes host access exclusive. Backends keep one per host-visible
/// buffer.
#[derive(Clone, Debug, Default)]
pub struct HostAccessLock(Arc<AtomicBool>);

impl HostAccessLock {
    /// A free lock.
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes the lock, or returns `None` while a guard holds it.
    pub fn try_acquire(&self) -> Option<HostAccessGuard> {
        self.0
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| HostAccessGuard(Arc::clone(&self.0)))
    }

    /// Takes the lock for `what`, or reports that the buffer is mapped.
    ///
    /// # Errors
    /// `RhiError::ValidationError` while a guard holds the lock.
    pub fn acquire_for(&self, what: &str) -> Result<HostAccessGuard, RhiError> {
        self.try_acquire().ok_or_else(|| {
            RhiError::ValidationError(format!(
                "{what}: the buffer is mapped; drop its MappedBuffer first"
            ))
        })
    }
}

/// Holds a [`HostAccessLock`] until dropped.
#[derive(Debug)]
pub struct HostAccessGuard(Arc<AtomicBool>);

impl Drop for HostAccessGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Exclusive host access to a mapped buffer (see the module documentation).
pub struct MappedBuffer<'a> {
    ptr: NonNull<u8>,
    len: u64,
    _guard: HostAccessGuard,
    _device: PhantomData<&'a ()>,
}

// SAFETY: the guard makes host access to the memory exclusive, and the memory is plain process
// memory that any thread may access.
unsafe impl Send for MappedBuffer<'_> {}
// SAFETY: shared references only allow copying data out, which may happen concurrently.
unsafe impl Sync for MappedBuffer<'_> {}

impl std::fmt::Debug for MappedBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedBuffer")
            .field("len", &self.len)
            .finish()
    }
}

impl<'a> MappedBuffer<'a> {
    /// Wraps `len` bytes of mapped memory at `ptr`.
    ///
    /// # Safety
    /// `ptr` must be valid for reads and writes of `len` bytes for as long as `guard` is held
    /// and the device borrowed for `'a` lives; the backend must not unmap or free the memory
    /// while `guard`'s lock is taken.
    pub unsafe fn new(ptr: NonNull<u8>, len: u64, guard: HostAccessGuard) -> Self {
        Self {
            ptr,
            len,
            _guard: guard,
            _device: PhantomData,
        }
    }

    /// Size of the mapping in bytes (the buffer's size).
    pub fn len(&self) -> u64 {
        self.len
    }

    /// True for an empty buffer.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn check(&self, offset: u64, size: usize) -> Result<usize, RhiError> {
        match offset.checked_add(size as u64) {
            Some(end) if end <= self.len => Ok(offset as usize),
            _ => Err(RhiError::ValidationError(format!(
                "range {offset}+{size} is outside the {}-byte mapping",
                self.len
            ))),
        }
    }

    /// Copies `data` into the buffer at `offset`.
    ///
    /// # Errors
    /// `RhiError::ValidationError` if the range is outside the buffer.
    pub fn write(&mut self, offset: u64, data: &[u8]) -> Result<(), RhiError> {
        let start = self.check(offset, data.len())?;
        // SAFETY: the range lies within the mapping (checked), which is valid while the guard
        // is held; `data` cannot overlap device memory the guard owns exclusively.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr().add(start), data.len());
        }
        Ok(())
    }

    /// Copies `out.len()` bytes from `offset` into `out`.
    ///
    /// # Errors
    /// `RhiError::ValidationError` if the range is outside the buffer.
    pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), RhiError> {
        let start = self.check(offset, out.len())?;
        // SAFETY: as in `write`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr().add(start),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }

    /// The start of the mapping, for writing directly (valid for `len()` bytes while `self`
    /// lives). Dereferencing it is up to the caller.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_access_is_exclusive_until_the_guard_drops() {
        let lock = HostAccessLock::new();
        let guard = lock.try_acquire().expect("free");
        assert!(lock.try_acquire().is_none());
        assert!(lock.acquire_for("write_buffer").is_err());
        drop(guard);
        assert!(lock.try_acquire().is_some());
    }

    #[test]
    fn mapped_ranges_are_bounds_checked() {
        let mut memory = [0_u8; 16];
        let lock = HostAccessLock::new();
        let ptr = NonNull::new(memory.as_mut_ptr()).unwrap();
        // SAFETY: `memory` outlives the mapping, which is dropped below.
        let mut mapped = unsafe { MappedBuffer::new(ptr, 16, lock.try_acquire().unwrap()) };
        mapped.write(12, &[1, 2, 3, 4]).expect("in bounds");
        assert!(mapped.write(13, &[1, 2, 3, 4]).is_err());
        let mut out = [0; 2];
        mapped.read(13, &mut out).expect("in bounds");
        assert_eq!(out, [2, 3]);
        assert!(mapped.read(u64::MAX, &mut out).is_err());
        drop(mapped);
        assert_eq!(memory[12..], [1, 2, 3, 4]);
    }
}
