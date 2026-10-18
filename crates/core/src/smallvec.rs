//! Inline-first small vector (`SmallVec`) and constructor macro (`smallvec!`) (`Rule 2`, `FIX-11.2`).
//!
//! # Purpose
//! Stores up to `A::CAPACITY` elements directly inline inside the struct without heap allocation,
//! spilling to a heap-allocated `Vec` only when the length exceeds the inline capacity. Used
//! throughout `korppi_rhi` and `korppi_render` to keep render graph compilation and RHI command
//! recording allocation-free in the steady state.
//!
//! # Inputs / Outputs
//! Parameterized by a fixed-size array type `[T; N]` implementing [`Array`]. Exposes a `Vec`-like
//! API (`new`, `push`, `pop`, `insert`, `remove`, `swap_remove`, `truncate`, `clear`, `resize`,
//! `dedup`, `retain`, `extend_from_slice`, `into_vec`, and `Deref<Target = [T]>`).

use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FromIterator;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr;
use core::slice::{self, SliceIndex};

/// Trait implemented by fixed-size arrays `[T; N]` that define the inline storage layout of a
/// [`SmallVec`].
///
/// # Safety
/// Implementers must be fixed-size contiguous arrays `[Self::Item; Self::CAPACITY]` whose memory
/// layout in `MaybeUninit<Self>` is identical to `[MaybeUninit<Self::Item>; Self::CAPACITY]`.
pub unsafe trait Array: Sized {
    /// Element type stored in the array.
    type Item;
    /// Number of elements that fit inline without heap allocation.
    const CAPACITY: usize;
}

// SAFETY: `[T; N]` is a contiguous array of `N` elements of type `T`, so `MaybeUninit<[T; N]>`
// has the exact layout and alignment required to hold up to `N` inline `T` values.
unsafe impl<T, const N: usize> Array for [T; N] {
    type Item = T;
    const CAPACITY: usize = N;
}

enum Storage<A: Array> {
    Inline { len: usize, buf: MaybeUninit<A> },
    Heap(Vec<A::Item>),
}

/// A vector that stores up to `A::CAPACITY` elements inline on the stack/struct and spills to a
/// heap allocation only when its length grows past `A::CAPACITY`.
pub struct SmallVec<A: Array> {
    storage: Storage<A>,
}

// SAFETY: `SmallVec<A>` logically owns `A::Item` elements (either inline in `MaybeUninit<A>` or
// on the heap in `Vec<A::Item>`), so it is `Send` whenever `A::Item: Send`.
unsafe impl<A: Array> Send for SmallVec<A> where A::Item: Send {}

// SAFETY: Shared references to `SmallVec<A>` only expose `&[A::Item]`, so it is `Sync` whenever
// `A::Item: Sync`.
unsafe impl<A: Array> Sync for SmallVec<A> where A::Item: Sync {}

impl<A: Array> SmallVec<A> {
    /// Creates an empty `SmallVec` backed by inline storage without allocating on the heap.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            storage: Storage::Inline {
                len: 0,
                buf: MaybeUninit::uninit(),
            },
        }
    }

    /// Creates an empty `SmallVec` with at least the specified capacity, spilling to the heap
    /// only if `cap > A::CAPACITY`.
    #[inline]
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        if cap <= A::CAPACITY {
            Self::new()
        } else {
            Self {
                storage: Storage::Heap(Vec::with_capacity(cap)),
            }
        }
    }

    /// Creates a `SmallVec` by cloning elements from `slice`.
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &[A::Item]) -> Self
    where
        A::Item: Clone,
    {
        let mut sv = Self::with_capacity(slice.len());
        sv.extend_from_slice(slice);
        sv
    }

    /// Creates a `SmallVec` directly from a full inline array `buf` without heap allocation.
    #[inline]
    #[must_use]
    pub const fn from_buf(buf: A) -> Self {
        Self {
            storage: Storage::Inline {
                len: A::CAPACITY,
                buf: MaybeUninit::new(buf),
            },
        }
    }

    /// Returns the number of elements in the vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Inline { len, .. } => *len,
            Storage::Heap(v) => v.len(),
        }
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the current capacity (at least `A::CAPACITY` when inline, or the heap capacity when
    /// spilled).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        match &self.storage {
            Storage::Inline { .. } => A::CAPACITY,
            Storage::Heap(v) => v.capacity(),
        }
    }

    /// Returns the maximum number of elements that can be stored inline without heap allocation.
    #[inline]
    #[must_use]
    pub const fn inline_size(&self) -> usize {
        A::CAPACITY
    }

    /// Returns `true` if the elements have spilled to a heap allocation.
    #[inline]
    #[must_use]
    pub fn spilled(&self) -> bool {
        matches!(self.storage, Storage::Heap(_))
    }

    /// Returns a raw const pointer to the vector's buffer.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const A::Item {
        match &self.storage {
            Storage::Inline { buf, .. } => buf.as_ptr().cast::<A::Item>(),
            Storage::Heap(v) => v.as_ptr(),
        }
    }

    /// Returns a raw mutable pointer to the vector's buffer.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut A::Item {
        match &mut self.storage {
            Storage::Inline { buf, .. } => buf.as_mut_ptr().cast::<A::Item>(),
            Storage::Heap(v) => v.as_mut_ptr(),
        }
    }

    /// Returns a shared slice over all initialized elements.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[A::Item] {
        match &self.storage {
            Storage::Inline { len, buf } => {
                let ptr = buf.as_ptr().cast::<A::Item>();
                // SAFETY: In the `Inline` state, the first `*len` elements of `buf` (where
                // `*len <= A::CAPACITY`) are initialized and valid for the shared borrow of `self`.
                unsafe { slice::from_raw_parts(ptr, *len) }
            }
            Storage::Heap(v) => v.as_slice(),
        }
    }

    /// Returns a mutable slice over all initialized elements.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [A::Item] {
        match &mut self.storage {
            Storage::Inline { len, buf } => {
                let ptr = buf.as_mut_ptr().cast::<A::Item>();
                // SAFETY: In the `Inline` state, the first `*len` elements of `buf` (where
                // `*len <= A::CAPACITY`) are initialized and uniquely borrowed via `&mut self`.
                unsafe { slice::from_raw_parts_mut(ptr, *len) }
            }
            Storage::Heap(v) => v.as_mut_slice(),
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        match &mut self.storage {
            Storage::Inline { len, buf } => {
                let required = len
                    .checked_add(additional)
                    .expect("SmallVec::reserve capacity overflow");
                if required > A::CAPACITY {
                    let current_len = *len;
                    *len = 0;
                    let mut vec = Vec::with_capacity(required.max(A::CAPACITY.saturating_mul(2)));
                    let src = buf.as_ptr().cast::<A::Item>();
                    let dst = vec.as_mut_ptr();
                    // SAFETY: `current_len` elements in `buf` were initialized and ownership is
                    // transferred to `vec` (`*len` was reset to 0 above so inline drop will not
                    // double-drop them). `vec` has capacity `>= required >= current_len`, and
                    // `src` and `dst` do not overlap.
                    unsafe {
                        ptr::copy_nonoverlapping(src, dst, current_len);
                        vec.set_len(current_len);
                    }
                    self.storage = Storage::Heap(vec);
                }
            }
            Storage::Heap(v) => v.reserve(additional),
        }
    }

    /// Appends an element to the back of the vector.
    #[inline]
    pub fn push(&mut self, value: A::Item) {
        if let Storage::Inline { len, buf } = &mut self.storage {
            if *len < A::CAPACITY {
                let dst = buf.as_mut_ptr().cast::<A::Item>();
                // SAFETY: `*len < A::CAPACITY`, so `dst.add(*len)` points to an uninitialized slot
                // within the bounds of `MaybeUninit<A>`. Writing `value` initializes that slot
                // before incrementing `*len`.
                unsafe {
                    dst.add(*len).write(value);
                }
                *len += 1;
                return;
            }
        }
        self.reserve(1);
        match &mut self.storage {
            Storage::Heap(v) => v.push(value),
            Storage::Inline { .. } => unreachable!("reserve(1) at full inline capacity spills"),
        }
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<A::Item> {
        match &mut self.storage {
            Storage::Inline { len, buf } => {
                if *len == 0 {
                    None
                } else {
                    *len -= 1;
                    let src = buf.as_ptr().cast::<A::Item>();
                    // SAFETY: The slot at `*len` was initialized prior to decrementing `*len`, and
                    // decrementing `*len` transfers ownership of that element out of `buf`.
                    Some(unsafe { src.add(*len).read() })
                }
            }
            Storage::Heap(v) => v.pop(),
        }
    }

    /// Shortens the vector, keeping the first `new_len` elements and dropping the rest.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        match &mut self.storage {
            Storage::Inline { len, buf } => {
                if new_len < *len {
                    let old_len = *len;
                    *len = new_len;
                    let base = buf.as_mut_ptr().cast::<A::Item>();
                    // SAFETY: Elements in `new_len..old_len` were initialized, and `*len` has
                    // already been lowered to `new_len` so they will not be dropped again if a
                    // destructor panics.
                    unsafe {
                        let tail =
                            ptr::slice_from_raw_parts_mut(base.add(new_len), old_len - new_len);
                        ptr::drop_in_place(tail);
                    }
                }
            }
            Storage::Heap(v) => v.truncate(new_len),
        }
    }

    /// Clears the vector, removing and dropping all values while preserving allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Inserts an element at position `index` within the vector, shifting all elements after it
    /// to the right.
    ///
    /// # Panics
    /// Panics if `index > len`.
    #[inline]
    pub fn insert(&mut self, index: usize, element: A::Item) {
        let current_len = self.len();
        assert!(
            index <= current_len,
            "SmallVec::insert index ({index}) out of bounds for length ({current_len})"
        );
        if current_len == self.capacity() {
            self.reserve(1);
        }
        match &mut self.storage {
            Storage::Inline { len, buf } => {
                let base = buf.as_mut_ptr().cast::<A::Item>();
                // SAFETY: `index <= *len < A::CAPACITY` (since `reserve(1)` above spilled if
                // `*len == A::CAPACITY`). Shifting `*len - index` elements right by 1 stays within
                // `A::CAPACITY`, and writing `element` at `index` leaves `0..(*len + 1)`
                // initialized.
                unsafe {
                    if index < *len {
                        ptr::copy(base.add(index), base.add(index + 1), *len - index);
                    }
                    base.add(index).write(element);
                }
                *len += 1;
            }
            Storage::Heap(v) => v.insert(index, element),
        }
    }

    /// Removes and returns the element at position `index`, shifting all elements after it to the
    /// left.
    ///
    /// # Panics
    /// Panics if `index >= len`.
    #[inline]
    pub fn remove(&mut self, index: usize) -> A::Item {
        match &mut self.storage {
            Storage::Inline { len, buf } => {
                assert!(
                    index < *len,
                    "SmallVec::remove index ({index}) out of bounds for length ({})",
                    *len
                );
                *len -= 1;
                let base = buf.as_mut_ptr().cast::<A::Item>();
                // SAFETY: `index <= *len` (after decrement), so `base.add(index)` is initialized.
                // Reading it moves out the value, and `ptr::copy` shifts the remaining `*len - index`
                // initialized elements down by one slot.
                unsafe {
                    let removed = base.add(index).read();
                    if index < *len {
                        ptr::copy(base.add(index + 1), base.add(index), *len - index);
                    }
                    removed
                }
            }
            Storage::Heap(v) => v.remove(index),
        }
    }

    /// Removes an element from the vector and returns it, replacing it with the last element.
    ///
    /// # Panics
    /// Panics if `index >= len`.
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> A::Item {
        match &mut self.storage {
            Storage::Inline { len, buf } => {
                assert!(
                    index < *len,
                    "SmallVec::swap_remove index ({index}) out of bounds for length ({})",
                    *len
                );
                *len -= 1;
                let base = buf.as_mut_ptr().cast::<A::Item>();
                // SAFETY: Both `index` and `*len` (the former last element) are initialized slots
                // within `0..=*len`. Reading `index` and copying the last slot into `index` when
                // `index != *len` leaves `0..*len` initialized.
                unsafe {
                    let removed = base.add(index).read();
                    if index != *len {
                        let last = base.add(*len).read();
                        base.add(index).write(last);
                    }
                    removed
                }
            }
            Storage::Heap(v) => v.swap_remove(index),
        }
    }

    /// Resizes the `SmallVec` in-place so that `len` is equal to `new_len`.
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: A::Item)
    where
        A::Item: Clone,
    {
        let current_len = self.len();
        if new_len <= current_len {
            self.truncate(new_len);
        } else {
            let additional = new_len - current_len;
            self.reserve(additional);
            for _ in 0..(additional - 1) {
                self.push(value.clone());
            }
            self.push(value);
        }
    }

    /// Clones and appends all elements in `other` to the vector.
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[A::Item])
    where
        A::Item: Clone,
    {
        self.reserve(other.len());
        for item in other {
            self.push(item.clone());
        }
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to each.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut A::Item) -> bool,
    {
        match &mut self.storage {
            Storage::Inline { .. } => {
                let mut i = 0;
                while i < self.len() {
                    if !f(&mut self[i]) {
                        self.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
            Storage::Heap(v) => v.retain_mut(f),
        }
    }

    /// Removes consecutive repeated elements in the vector according to [`PartialEq`].
    #[inline]
    pub fn dedup(&mut self)
    where
        A::Item: PartialEq,
    {
        match &mut self.storage {
            Storage::Inline { .. } => {
                let mut i = 1;
                while i < self.len() {
                    if self[i - 1] == self[i] {
                        self.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
            Storage::Heap(v) => v.dedup(),
        }
    }

    /// Converts the `SmallVec` into a standard heap-allocated [`Vec`], reusing the existing heap
    /// buffer in $O(1)$ time if already spilled.
    #[inline]
    #[must_use]
    pub fn into_vec(mut self) -> Vec<A::Item> {
        let taken = core::mem::replace(
            &mut self.storage,
            Storage::Inline {
                len: 0,
                buf: MaybeUninit::uninit(),
            },
        );
        match taken {
            Storage::Inline { len, buf } => {
                let mut vec = Vec::with_capacity(len);
                let src = buf.as_ptr().cast::<A::Item>();
                let dst = vec.as_mut_ptr();
                // SAFETY: `len` elements in `buf` were initialized and moved out of `self` by
                // replacing `self.storage` with `len: 0`. `vec` has capacity `len` and does not
                // overlap `buf`.
                unsafe {
                    ptr::copy_nonoverlapping(src, dst, len);
                    vec.set_len(len);
                }
                vec
            }
            Storage::Heap(v) => v,
        }
    }
}

impl<A: Array> Drop for SmallVec<A> {
    #[inline]
    fn drop(&mut self) {
        if let Storage::Inline { len, buf } = &mut self.storage {
            let count = *len;
            *len = 0;
            let ptr = buf.as_mut_ptr().cast::<A::Item>();
            // SAFETY: The first `count` elements in `buf` were initialized, and `*len` was reset
            // to 0 before calling `drop_in_place`.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(ptr, count));
            }
        }
    }
}

impl<A: Array> Default for SmallVec<A> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Array> Deref for SmallVec<A> {
    type Target = [A::Item];

    #[inline]
    fn deref(&self) -> &[A::Item] {
        self.as_slice()
    }
}

impl<A: Array> DerefMut for SmallVec<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [A::Item] {
        self.as_mut_slice()
    }
}

impl<A: Array, I: SliceIndex<[A::Item]>> Index<I> for SmallVec<A> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<A: Array, I: SliceIndex<[A::Item]>> IndexMut<I> for SmallVec<A> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_mut_slice(), index)
    }
}

impl<A: Array> AsRef<[A::Item]> for SmallVec<A> {
    #[inline]
    fn as_ref(&self) -> &[A::Item] {
        self.as_slice()
    }
}

impl<A: Array> AsMut<[A::Item]> for SmallVec<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [A::Item] {
        self.as_mut_slice()
    }
}

impl<A: Array> Borrow<[A::Item]> for SmallVec<A> {
    #[inline]
    fn borrow(&self) -> &[A::Item] {
        self.as_slice()
    }
}

impl<A: Array> BorrowMut<[A::Item]> for SmallVec<A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [A::Item] {
        self.as_mut_slice()
    }
}

impl<A: Array> Clone for SmallVec<A>
where
    A::Item: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl<A: Array> fmt::Debug for SmallVec<A>
where
    A::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<A: Array, B: Array> PartialEq<SmallVec<B>> for SmallVec<A>
where
    A::Item: PartialEq<B::Item>,
{
    #[inline]
    fn eq(&self, other: &SmallVec<B>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<A: Array> Eq for SmallVec<A> where A::Item: Eq {}

impl<A: Array> PartialOrd for SmallVec<A>
where
    A::Item: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.as_slice(), other.as_slice())
    }
}

impl<A: Array> Ord for SmallVec<A>
where
    A::Item: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self.as_slice(), other.as_slice())
    }
}

impl<A: Array> Hash for SmallVec<A>
where
    A::Item: Hash,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<A: Array> From<&[A::Item]> for SmallVec<A>
where
    A::Item: Clone,
{
    #[inline]
    fn from(slice: &[A::Item]) -> Self {
        Self::from_slice(slice)
    }
}

impl<A: Array> From<Vec<A::Item>> for SmallVec<A> {
    #[inline]
    fn from(vec: Vec<A::Item>) -> Self {
        if vec.len() <= A::CAPACITY {
            let mut sv = Self::new();
            for item in vec {
                sv.push(item);
            }
            sv
        } else {
            Self {
                storage: Storage::Heap(vec),
            }
        }
    }
}

impl<A: Array> From<A> for SmallVec<A> {
    #[inline]
    fn from(array: A) -> Self {
        Self::from_buf(array)
    }
}

impl<A: Array> Extend<A::Item> for SmallVec<A> {
    #[inline]
    fn extend<I: IntoIterator<Item = A::Item>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(lower);
        for item in iter {
            self.push(item);
        }
    }
}

impl<A: Array> FromIterator<A::Item> for SmallVec<A> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = A::Item>>(iter: I) -> Self {
        let mut sv = Self::new();
        sv.extend(iter);
        sv
    }
}

enum IntoIterStorage<A: Array> {
    Inline {
        buf: MaybeUninit<A>,
        pos: usize,
        end: usize,
    },
    Heap(std::vec::IntoIter<A::Item>),
}

/// By-value iterator over a [`SmallVec`].
pub struct IntoIter<A: Array> {
    storage: IntoIterStorage<A>,
}

// SAFETY: `IntoIter<A>` owns remaining `A::Item` values, so it is `Send` when `A::Item: Send`.
unsafe impl<A: Array> Send for IntoIter<A> where A::Item: Send {}
// SAFETY: Shared references to `IntoIter<A>` do not allow unsynchronized mutation, so it is `Sync`
// when `A::Item: Sync`.
unsafe impl<A: Array> Sync for IntoIter<A> where A::Item: Sync {}

impl<A: Array> Iterator for IntoIter<A> {
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.storage {
            IntoIterStorage::Inline { buf, pos, end } => {
                if *pos < *end {
                    let idx = *pos;
                    *pos += 1;
                    let base = buf.as_ptr().cast::<A::Item>();
                    // SAFETY: `idx < *end <= A::CAPACITY` was initialized when the `IntoIter` was
                    // constructed, and incrementing `*pos` ensures `idx` is read at most once and
                    // skipped by `Drop`.
                    Some(unsafe { base.add(idx).read() })
                } else {
                    None
                }
            }
            IntoIterStorage::Heap(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.len();
        (rem, Some(rem))
    }
}

impl<A: Array> DoubleEndedIterator for IntoIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.storage {
            IntoIterStorage::Inline { buf, pos, end } => {
                if *pos < *end {
                    *end -= 1;
                    let base = buf.as_ptr().cast::<A::Item>();
                    // SAFETY: `*end` (after decrement) was initialized and is now outside
                    // `*pos..*end`, so it is read once and not dropped again.
                    Some(unsafe { base.add(*end).read() })
                } else {
                    None
                }
            }
            IntoIterStorage::Heap(iter) => iter.next_back(),
        }
    }
}

impl<A: Array> ExactSizeIterator for IntoIter<A> {
    #[inline]
    fn len(&self) -> usize {
        match &self.storage {
            IntoIterStorage::Inline { pos, end, .. } => *end - *pos,
            IntoIterStorage::Heap(iter) => iter.len(),
        }
    }
}

impl<A: Array> Drop for IntoIter<A> {
    #[inline]
    fn drop(&mut self) {
        if let IntoIterStorage::Inline { buf, pos, end } = &mut self.storage {
            let start = *pos;
            let count = *end - start;
            *pos = *end;
            if count > 0 {
                let base = buf.as_mut_ptr().cast::<A::Item>();
                // SAFETY: Elements in `start..*end` have not yet been yielded by `next` or
                // `next_back`, and `*pos = *end` prevents double-dropping on panic.
                unsafe {
                    ptr::drop_in_place(ptr::slice_from_raw_parts_mut(base.add(start), count));
                }
            }
        }
    }
}

impl<A: Array> IntoIterator for SmallVec<A> {
    type Item = A::Item;
    type IntoIter = IntoIter<A>;

    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        let taken = core::mem::replace(
            &mut self.storage,
            Storage::Inline {
                len: 0,
                buf: MaybeUninit::uninit(),
            },
        );
        let storage = match taken {
            Storage::Inline { len, buf } => IntoIterStorage::Inline {
                buf,
                pos: 0,
                end: len,
            },
            Storage::Heap(v) => IntoIterStorage::Heap(v.into_iter()),
        };
        IntoIter { storage }
    }
}

impl<'a, A: Array> IntoIterator for &'a SmallVec<A> {
    type Item = &'a A::Item;
    type IntoIter = slice::Iter<'a, A::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, A: Array> IntoIterator for &'a mut SmallVec<A> {
    type Item = &'a mut A::Item;
    type IntoIter = slice::IterMut<'a, A::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

/// Constructs a [`SmallVec`] containing the given elements.
#[macro_export]
macro_rules! smallvec {
    () => {
        $crate::SmallVec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut sv = $crate::SmallVec::new();
        sv.resize($n, $elem);
        sv
    }};
    ($($x:expr),+ $(,)?) => {{
        let mut sv = $crate::SmallVec::new();
        $(
            sv.push($x);
        )+
        sv
    }};
}
