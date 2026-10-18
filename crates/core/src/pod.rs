//! Plain-Old-Data (`Pod`) and zero-initialization (`Zeroable`) traits and byte-casting helpers
//! (`Rule 2`, `FIX-11.2`).
//!
//! # Purpose
//! Provides safe, zero-cost byte reinterpretation (`bytes_of`, `cast_slice`, `pod_read_unaligned`)
//! for GPU-facing buffers, uniform blocks, push constants, and vertex/index streams without
//! external dependencies.
//!
//! # Inputs / Outputs
//! - Inputs: references or slices of types implementing [`Pod`] (primitive scalars, arrays of
//!   [`Pod`] types, and `#[repr(C)]` structs verified via [`crate::impl_pod!`]).
//! - Outputs: `&[u8]` / `&mut [u8]` byte views, reinterpreted `&[B]` / `&mut [B]` slices, or
//!   unaligned `T` values read from byte slices.

use core::mem::{align_of, size_of, MaybeUninit};
use core::slice;

/// Marker trait for types that can be safely initialized with all-zero bytes (`0u8`).
///
/// # Safety
/// Implementers must guarantee that a value of `Self` whose underlying memory representation
/// consists entirely of `0u8` bytes is a valid, initialized value of `Self` (for example, it
/// must not contain non-null references, `NonZero*` primitives, or enums without a zero
/// discriminant).
pub unsafe trait Zeroable: Sized {
    /// Returns a new value of `Self` with all bytes initialized to zero.
    #[inline]
    #[must_use]
    fn zeroed() -> Self {
        let uninit = MaybeUninit::<Self>::zeroed();
        // SAFETY: The `Zeroable` trait contract guarantees that an all-zero bit pattern is a valid
        // representation of `Self`.
        unsafe { uninit.assume_init() }
    }
}

/// Marker trait for Plain Old Data types that can be safely reinterpreted to and from arbitrary
/// byte slices.
///
/// # Safety
/// Implementers must guarantee that:
/// 1. The type has no uninitialized padding bytes (every byte between `0..size_of::<Self>()` is
///    part of an initialized field).
/// 2. Any bit pattern of length `size_of::<Self>()` is a valid value of `Self` (no `bool`, `char`,
///    enums with restricted discriminants, references, or pointers).
/// 3. The type is `Copy + 'static` and contains no interior mutability (`UnsafeCell`).
pub unsafe trait Pod: Zeroable + Copy + 'static {}

macro_rules! unsafe_impl_zeroable_and_pod {
    ($($ty:ty),* $(,)?) => {
        $(
            // SAFETY: Primitive numeric and unit types are valid when all bytes are zero.
            unsafe impl Zeroable for $ty {}
            // SAFETY: Primitive numeric and unit types are `Copy + 'static`, have no padding
            // bytes, and accept any bit pattern.
            unsafe impl Pod for $ty {}
        )*
    };
}

unsafe_impl_zeroable_and_pod!(
    (),
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    usize,
    isize,
    f32,
    f64,
);

// SAFETY: `bool` is represented as a single byte where `0u8` (`false`) is valid. (`bool` is not
// `Pod` because bit patterns > 1 are invalid.)
unsafe impl Zeroable for bool {}

// SAFETY: A fixed-size array `[T; N]` has the exact contiguous layout of `N` elements of `T` with
// no inter-element padding beyond `T` itself. If `T: Zeroable`, all-zero bytes produce `N` valid
// zeroed `T` elements.
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

// SAFETY: A fixed-size array `[T; N]` of `T: Pod` has no padding bytes, is `Copy + 'static`, and
// any bit pattern is valid for all `N` elements.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// Implements [`Zeroable`] and [`Pod`] for `#[repr(C)]` structs whose field sizes sum to the
/// expected byte size (proving at compile time that the struct contains no padding bytes).
#[macro_export]
macro_rules! impl_pod {
    ($($ty:ty = $field_bytes:expr),* $(,)?) => {$(
        const _: () = assert!(::core::mem::size_of::<$ty>() == $field_bytes);
        // SAFETY: The caller guarantees `$ty` is `#[repr(C)]` composed of `Pod` scalar/array
        // fields whose sizes sum to `$field_bytes`, which the compile-time assertion above proves
        // equals `size_of::<$ty>()` (so there are no padding bytes and all-zero is valid).
        unsafe impl $crate::pod::Zeroable for $ty {}
        // SAFETY: As above, `$ty` has no padding bytes, accepts any bit pattern, and is
        // `Copy + 'static` (enforced by the `Pod` supertrait bound).
        unsafe impl $crate::pod::Pod for $ty {}
    )*};
}

/// Reinterprets a shared reference to `T` as a byte slice `&[u8]` of length `size_of::<T>()`.
#[inline]
#[must_use]
pub fn bytes_of<T: Pod>(val: &T) -> &[u8] {
    let ptr = (val as *const T).cast::<u8>();
    let len = size_of::<T>();
    // SAFETY: `T: Pod` guarantees `val` has no uninitialized padding bytes and no interior
    // mutability, `u8` has alignment 1, and `ptr` is valid for `len` bytes for the lifetime of
    // `val`.
    unsafe { slice::from_raw_parts(ptr, len) }
}

/// Reinterprets a mutable reference to `T` as a mutable byte slice `&mut [u8]` of length
/// `size_of::<T>()`.
#[inline]
#[must_use]
pub fn bytes_of_mut<T: Pod>(val: &mut T) -> &mut [u8] {
    let ptr = (val as *mut T).cast::<u8>();
    let len = size_of::<T>();
    // SAFETY: `T: Pod` guarantees `val` has no padding bytes and accepts any bit pattern written
    // through the returned `&mut [u8]` slice, `u8` has alignment 1, and `ptr` is uniquely borrowed
    // for the lifetime of `val`.
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}

/// Reinterprets a slice `&[A]` as `&[B]` where both `A` and `B` implement [`Pod`].
///
/// # Panics
/// Panics if:
/// - one of `A` or `B` is zero-sized and the other is not (when `slice` is non-empty);
/// - the byte length `slice.len() * size_of::<A>()` is not an exact multiple of `size_of::<B>()`;
/// - `slice.as_ptr()` is not aligned to `align_of::<B>()` (when the resulting slice is non-empty).
#[inline]
#[must_use]
pub fn cast_slice<A: Pod, B: Pod>(slice: &[A]) -> &[B] {
    if size_of::<A>() == size_of::<B>() && align_of::<A>() == align_of::<B>() {
        let ptr = slice.as_ptr().cast::<B>();
        // SAFETY: `A` and `B` are both `Pod` with identical size and alignment, so `slice.len()`
        // elements of `A` are valid as `slice.len()` elements of `B`.
        return unsafe { slice::from_raw_parts(ptr, slice.len()) };
    }

    let input_bytes = slice
        .len()
        .checked_mul(size_of::<A>())
        .expect("cast_slice: byte length overflow");

    if input_bytes == 0 {
        return &[];
    }

    assert!(
        size_of::<B>() != 0,
        "cast_slice: cannot cast non-empty byte range into zero-sized type"
    );
    assert!(
        input_bytes.is_multiple_of(size_of::<B>()),
        "cast_slice: input byte length ({input_bytes}) is not a multiple of target element size ({})",
        size_of::<B>()
    );
    let addr = slice.as_ptr() as usize;
    assert!(
        addr.is_multiple_of(align_of::<B>()),
        "cast_slice: input pointer ({addr:#x}) is not aligned to target alignment ({})",
        align_of::<B>()
    );

    let out_len = input_bytes / size_of::<B>();
    let ptr = slice.as_ptr().cast::<B>();
    // SAFETY: Both `A` and `B` are `Pod` (no padding bytes, any bit pattern is valid), `ptr` is
    // aligned to `align_of::<B>()` (asserted above), and `out_len * size_of::<B>() == input_bytes`
    // matches the initialized byte range of `slice`.
    unsafe { slice::from_raw_parts(ptr, out_len) }
}

/// Reinterprets a mutable slice `&mut [A]` as `&mut [B]` where both `A` and `B` implement [`Pod`].
///
/// # Panics
/// Panics under the same size or alignment mismatch conditions as [`cast_slice`].
#[inline]
#[must_use]
pub fn cast_slice_mut<A: Pod, B: Pod>(slice: &mut [A]) -> &mut [B] {
    if size_of::<A>() == size_of::<B>() && align_of::<A>() == align_of::<B>() {
        let len = slice.len();
        let ptr = slice.as_mut_ptr().cast::<B>();
        // SAFETY: `A` and `B` are both `Pod` with identical size and alignment, so `len` elements
        // of `A` can be mutably viewed as `len` elements of `B` without violating validity.
        return unsafe { slice::from_raw_parts_mut(ptr, len) };
    }

    let input_bytes = slice
        .len()
        .checked_mul(size_of::<A>())
        .expect("cast_slice_mut: byte length overflow");

    if input_bytes == 0 {
        return &mut [];
    }

    assert!(
        size_of::<B>() != 0,
        "cast_slice_mut: cannot cast non-empty byte range into zero-sized type"
    );
    assert!(
        input_bytes.is_multiple_of(size_of::<B>()),
        "cast_slice_mut: input byte length ({input_bytes}) is not a multiple of target element size ({})",
        size_of::<B>()
    );
    let addr = slice.as_ptr() as usize;
    assert!(
        addr.is_multiple_of(align_of::<B>()),
        "cast_slice_mut: input pointer ({addr:#x}) is not aligned to target alignment ({})",
        align_of::<B>()
    );

    let out_len = input_bytes / size_of::<B>();
    let ptr = slice.as_mut_ptr().cast::<B>();
    // SAFETY: Both `A` and `B` are `Pod` (no padding bytes, any bit pattern is valid), `ptr` is
    // aligned to `align_of::<B>()` (asserted above), and `out_len * size_of::<B>() == input_bytes`
    // covers the exact mutable byte range of `slice`.
    unsafe { slice::from_raw_parts_mut(ptr, out_len) }
}

/// Reads a value of `T: Pod` from `bytes` without requiring `bytes` to be aligned to `T`.
///
/// # Panics
/// Panics if `bytes.len() != size_of::<T>()`.
#[inline]
#[must_use]
pub fn pod_read_unaligned<T: Pod>(bytes: &[u8]) -> T {
    assert_eq!(
        bytes.len(),
        size_of::<T>(),
        "pod_read_unaligned: byte slice length ({}) does not match target size ({})",
        bytes.len(),
        size_of::<T>()
    );
    let ptr = bytes.as_ptr().cast::<T>();
    // SAFETY: `bytes` has exact length `size_of::<T>()` (asserted above), `T: Pod` guarantees any
    // bit pattern of `size_of::<T>()` bytes is a valid `T` and `T: Copy`, and `read_unaligned`
    // does not require alignment of `ptr`.
    unsafe { core::ptr::read_unaligned(ptr) }
}
