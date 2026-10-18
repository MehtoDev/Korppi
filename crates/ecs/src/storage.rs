//! Type-erased contiguous columnar component storage (SoA).

use crate::component::ComponentInfo;
use crate::tick::{ComponentTicks, Tick};
use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr::NonNull;

/// Contiguous type-erased columnar memory for storing instances of a single component type.
///
/// Each row also records the [`ComponentTicks`] of its value, in an array kept in step with the
/// values. A mutable query that visits the whole column records its tick once for every row
/// (`bulk_changed`) instead of writing each row's tick; a row's changed tick is the newer of the
/// two. A row joining the column first settles the bulk change into the rows it covers.
pub struct Column {
    info: ComponentInfo,
    ptr: Option<NonNull<u8>>,
    capacity: usize,
    len: usize,
    ticks: Vec<ComponentTicks>,
    bulk_changed: Option<Tick>,
}

// SAFETY: All component types implement Component: 'static + Send + Sync, so Column can be safely sent across threads.
unsafe impl Send for Column {}
// SAFETY: All component types implement Component: 'static + Send + Sync, and access is controlled via exclusive or shared references.
unsafe impl Sync for Column {}

impl Column {
    /// Creates an empty column for storing the component described by `info`.
    #[must_use]
    pub fn new(info: ComponentInfo) -> Self {
        let is_zst = info.layout().size() == 0;
        Self {
            info,
            ptr: if is_zst {
                Some(
                    NonNull::new(std::ptr::without_provenance_mut(info.layout().align()))
                        .expect("Alignment must be non-zero"),
                )
            } else {
                None
            },
            capacity: if is_zst { usize::MAX } else { 0 },
            len: 0,
            ticks: Vec::new(),
            bulk_changed: None,
        }
    }

    /// Makes this empty column store the component `info` describes, keeping its buffer.
    ///
    /// # Panics
    /// If the column holds values or `info` has another layout.
    pub(crate) fn reset(&mut self, info: ComponentInfo) {
        assert!(self.len == 0, "reset of a non-empty column");
        assert_eq!(info.layout(), self.info.layout(), "reset to another layout");
        self.info = info;
        self.bulk_changed = None;
    }

    /// Returns the number of component instances stored in the column.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the column contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the component metadata for this column.
    #[must_use]
    pub const fn info(&self) -> ComponentInfo {
        self.info
    }

    fn item_layout(&self) -> Layout {
        self.info.layout()
    }

    fn grow(&mut self) {
        let item_layout = self.item_layout();
        if item_layout.size() == 0 {
            self.capacity = usize::MAX;
            if self.ptr.is_none() {
                self.ptr = Some(
                    NonNull::new(item_layout.align() as *mut u8)
                        .expect("Alignment must be non-zero"),
                );
            }
            return;
        }

        let new_capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity
                .checked_mul(2)
                .expect("Capacity overflow in ECS column")
        };

        let new_size = item_layout
            .size()
            .checked_mul(new_capacity)
            .expect("Capacity overflow in ECS column");
        let new_layout = Layout::from_size_align(new_size, item_layout.align())
            .expect("Invalid layout for column allocation");

        let new_ptr = if let Some(old_ptr) = self.ptr {
            // SAFETY: old_ptr was previously allocated with this exact layout and capacity.
            let old_layout = unsafe {
                Layout::from_size_align_unchecked(
                    item_layout.size() * self.capacity,
                    item_layout.align(),
                )
            };
            // SAFETY: old_ptr is a valid allocation; new_layout has non-zero size and valid alignment.
            let raw = unsafe { realloc(old_ptr.as_ptr(), old_layout, new_layout.size()) };
            NonNull::new(raw).expect("Reallocation failed for ECS column")
        } else {
            // SAFETY: new_layout has non-zero size and valid alignment.
            let raw = unsafe { alloc(new_layout) };
            NonNull::new(raw).expect("Allocation failed for ECS column")
        };

        self.ptr = Some(new_ptr);
        self.capacity = new_capacity;
    }

    /// Pushes a typed component into the column, transferring ownership; `tick` is its added
    /// and changed tick.
    ///
    /// # Safety
    /// `T` must match the component type associated with this column.
    pub unsafe fn push<T>(&mut self, value: T, tick: Tick) {
        debug_assert_eq!(Layout::new::<T>(), self.item_layout());
        if self.len == self.capacity {
            self.grow();
        }

        if let Some(ptr) = self.ptr {
            let item_size = self.item_layout().size();
            if item_size > 0 {
                // SAFETY: caller guarantees T matches column layout; ptr is allocated for capacity > len.
                unsafe {
                    let dst = ptr.as_ptr().add(self.len * item_size) as *mut T;
                    std::ptr::write(dst, value);
                }
            } else {
                std::mem::forget(value);
            }
        }
        self.settle_bulk_changed(tick);
        self.ticks.push(ComponentTicks::new(tick));
        self.len += 1;
    }

    /// Appends a value given as its bytes; `tick` is its added and changed tick.
    ///
    /// # Safety
    /// `bytes` must have the column's size and form a valid value of its component.
    pub(crate) unsafe fn push_bytes(&mut self, bytes: &[u8], tick: Tick) {
        debug_assert_eq!(bytes.len(), self.item_layout().size());
        if self.len == self.capacity {
            self.grow();
        }
        let size = self.item_layout().size();
        if size > 0 {
            let ptr = self.ptr.expect("Column data is unallocated").as_ptr();
            // SAFETY: the column has room for one more value at `len`; the sizes match.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(self.len * size), size)
            };
        }
        self.settle_bulk_changed(tick);
        self.ticks.push(ComponentTicks::new(tick));
        self.len += 1;
    }

    /// Drops the value at `row` and writes the one given as `bytes` in its place.
    ///
    /// # Safety
    /// As for [`push_bytes`](Self::push_bytes), and `row < len`.
    pub(crate) unsafe fn replace_bytes(&mut self, row: usize, bytes: &[u8], tick: Tick) {
        let size = self.item_layout().size();
        let ptr = self.ptr.expect("Column data is unallocated").as_ptr();
        // SAFETY: `row` holds a live value, dropped once and then overwritten with a valid one.
        unsafe {
            let slot = ptr.add(row * size);
            self.info.drop_in_place(slot);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), slot, size);
        }
        self.ticks[row].changed = tick;
    }

    /// The bytes of the value at `row`.
    ///
    /// # Safety
    /// `row < len`, and every byte of the column's values must be initialized (true for a
    /// runtime-defined component, whose values are only ever written as bytes).
    pub(crate) unsafe fn row_bytes(&self, row: usize) -> &[u8] {
        let size = self.item_layout().size();
        if size == 0 {
            return &[];
        }
        let ptr = self.ptr.expect("Column data is unallocated").as_ptr();
        // SAFETY: the caller guarantees `row < len` and initialized bytes.
        unsafe { std::slice::from_raw_parts(ptr.add(row * size), size) }
    }

    /// The bytes of the value at `row`, mutably.
    ///
    /// # Safety
    /// As for [`row_bytes`](Self::row_bytes), and any bytes written must keep the value valid.
    pub(crate) unsafe fn row_bytes_mut(&mut self, row: usize) -> &mut [u8] {
        let size = self.item_layout().size();
        if size == 0 {
            return &mut [];
        }
        let ptr = self.ptr.expect("Column data is unallocated").as_ptr();
        // SAFETY: as for `row_bytes`, through the exclusive borrow.
        unsafe { std::slice::from_raw_parts_mut(ptr.add(row * size), size) }
    }

    /// Returns a typed immutable reference to the component at `row`.
    ///
    /// # Safety
    /// `T` must match the component type and `row` must be `< self.len`.
    #[must_use]
    pub unsafe fn get<T>(&self, row: usize) -> &T {
        debug_assert!(row < self.len);
        let ptr = self.ptr.expect("Column data is unallocated").as_ptr();
        let item_size = self.item_layout().size();
        // SAFETY: caller guarantees T matches component type; row < len; ptr is valid (or aligned dangling for ZST).
        unsafe {
            let item_ptr = if item_size > 0 {
                ptr.add(row * item_size) as *const T
            } else {
                ptr as *const T
            };
            &*item_ptr
        }
    }

    /// Returns a typed mutable reference to the component at `row`.
    ///
    /// # Safety
    /// `T` must match the component type and `row` must be `< self.len`.
    pub unsafe fn get_mut<T>(&mut self, row: usize) -> &mut T {
        debug_assert!(row < self.len);
        let ptr = self.ptr.expect("Column data is unallocated").as_ptr();
        let item_size = self.item_layout().size();
        // SAFETY: caller guarantees T matches component type; row < len; ptr is valid (or aligned dangling for ZST).
        unsafe {
            let item_ptr = if item_size > 0 {
                ptr.add(row * item_size) as *mut T
            } else {
                ptr as *mut T
            };
            &mut *item_ptr
        }
    }

    /// Returns the raw pointer to the base of the column's data buffer.
    #[must_use]
    #[inline(always)]
    pub fn raw_ptr(&self) -> *mut u8 {
        self.ptr.map_or(std::ptr::null_mut(), NonNull::as_ptr)
    }

    /// Returns a typed immutable slice of the column's contiguous component elements.
    ///
    /// # Safety
    /// `T` must match the component type associated with this column.
    #[must_use]
    #[inline(always)]
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            let ptr = self.raw_ptr() as *const T;
            // SAFETY: caller guarantees T matches column component type; elements 0..self.len are initialized.
            unsafe { std::slice::from_raw_parts(ptr, self.len) }
        }
    }

    /// Returns a typed mutable slice of the column's contiguous component elements.
    ///
    /// # Safety
    /// `T` must match the component type associated with this column.
    #[must_use]
    #[inline(always)]
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            let ptr = self.raw_ptr() as *mut T;
            // SAFETY: caller guarantees T matches column component type; elements 0..self.len are initialized;
            // exclusive borrow ensures no aliasing.
            unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
        }
    }

    /// Removes the component at `row` by swapping it with the last element and dropping it.
    /// Returns `true` if a swap occurred (i.e. `row < self.len - 1`).
    pub fn swap_remove(&mut self, row: usize) -> bool {
        assert!(row < self.len);
        let item_size = self.item_layout().size();
        let last_idx = self.len - 1;
        let ptr = self.ptr.expect("Column data is unallocated").as_ptr();
        // SAFETY: `row` and `last_idx` are below `len`, so both slots hold live values inside the
        // allocation (for a zero-sized type `ptr` is aligned and dangling). The removed value is
        // swapped to the last slot, which the shrunk length then excludes, and dropped once.
        unsafe {
            if item_size > 0 && row != last_idx {
                std::ptr::swap_nonoverlapping(
                    ptr.add(row * item_size),
                    ptr.add(last_idx * item_size),
                    item_size,
                );
            }
            self.len = last_idx;
            self.info.drop_in_place(ptr.add(last_idx * item_size));
        }
        self.ticks.swap_remove(row);
        row != last_idx
    }

    /// Moves the component at `row` directly to the end of `dst` column via memory copy without dropping,
    /// and fills the vacant slot in `self` with its last element. The value keeps its ticks; `now`
    /// is the world's current tick, to tell whether this column's bulk change is the newer one.
    ///
    /// Returns `true` if an element in `self` was swapped into `row`.
    ///
    /// # Safety
    /// `self` and `dst` must describe identical component layouts.
    pub unsafe fn move_row_to(&mut self, row: usize, dst: &mut Self, now: Tick) -> bool {
        assert!(row < self.len);
        let item_size = self.item_layout().size();
        let last_idx = self.len - 1;

        if item_size > 0 {
            if dst.len == dst.capacity {
                dst.grow();
            }

            let src_ptr = self.ptr.expect("Source column unallocated").as_ptr();
            let dst_ptr = dst.ptr.expect("Destination column unallocated").as_ptr();

            // SAFETY: caller guarantees matching component layouts; row and last_idx are < self.len; dst.len < dst.capacity.
            unsafe {
                let src_row_ptr = src_ptr.add(row * item_size);
                let dst_row_ptr = dst_ptr.add(dst.len * item_size);

                // Copy element to destination
                std::ptr::copy_nonoverlapping(src_row_ptr, dst_row_ptr, item_size);

                // If not the last element, copy the last element into row
                if row != last_idx {
                    let last_ptr = src_ptr.add(last_idx * item_size);
                    std::ptr::copy_nonoverlapping(last_ptr, src_row_ptr, item_size);
                }
            }
        }

        let ticks = ComponentTicks {
            added: self.ticks[row].added,
            changed: self.changed_tick(row, now),
        };
        self.ticks.swap_remove(row);
        dst.settle_bulk_changed(now);
        dst.ticks.push(ticks);
        dst.len += 1;
        self.len -= 1;
        row != last_idx
    }

    /// Removes and transfers ownership of the component at `row` into `T` without dropping it,
    /// swapping with the last element.
    ///
    /// # Safety
    /// `T` must match the component type and `row` must be `< self.len`.
    pub unsafe fn swap_remove_and_take<T>(&mut self, row: usize) -> (T, bool) {
        assert!(row < self.len);
        let item_size = self.item_layout().size();
        let last_idx = self.len - 1;

        self.ticks.swap_remove(row);
        if item_size == 0 {
            self.len -= 1;
            // SAFETY: T is zero-sized; read from dangling aligned pointer is safe and creates a valid ZST value.
            let value = unsafe { std::ptr::read(NonNull::<T>::dangling().as_ptr()) };
            return (value, false);
        }

        let ptr = self.ptr.expect("Column data is unallocated").as_ptr();

        // SAFETY: caller guarantees T matches column component type; row < len; buffer is allocated.
        let (value, swapped) = unsafe {
            let row_ptr = ptr.add(row * item_size);
            let value = std::ptr::read(row_ptr as *const T);

            let swapped = if row != last_idx {
                let last_ptr = ptr.add(last_idx * item_size);
                std::ptr::copy_nonoverlapping(last_ptr, row_ptr, item_size);
                true
            } else {
                false
            };
            (value, swapped)
        };

        self.len -= 1;
        (value, swapped)
    }

    /// Clears all elements from the column and drops them.
    pub fn clear(&mut self) {
        self.ticks.clear();
        self.bulk_changed = None;
        let len = std::mem::replace(&mut self.len, 0);
        if self.info.needs_drop() {
            if let Some(ptr) = self.ptr {
                let item_size = self.item_layout().size();
                for i in 0..len {
                    // SAFETY: rows below the old length held live values, each dropped once here.
                    unsafe { self.info.drop_in_place(ptr.as_ptr().add(i * item_size)) };
                }
            }
        }
    }
}

impl Column {
    /// The added and own changed tick of each row.
    #[must_use]
    pub fn ticks(&self) -> &[ComponentTicks] {
        &self.ticks
    }

    /// The tick a mutable query last changed every row at, if one did.
    #[must_use]
    pub const fn bulk_changed(&self) -> Option<Tick> {
        self.bulk_changed
    }

    /// The tick the value at `row` last changed at, as seen from `now`.
    #[must_use]
    pub fn changed_tick(&self, row: usize, now: Tick) -> Tick {
        let own = self.ticks[row].changed;
        match self.bulk_changed {
            Some(bulk) if bulk.is_newer_than(own, now) => bulk,
            _ => own,
        }
    }

    /// Writes a pending bulk change into the rows it covers, before a row joins the column: the
    /// bulk change applies only to the rows present when it was made.
    fn settle_bulk_changed(&mut self, now: Tick) {
        if let Some(bulk) = self.bulk_changed.take() {
            for ticks in &mut self.ticks {
                if bulk.is_newer_than(ticks.changed, now) {
                    ticks.changed = bulk;
                }
            }
        }
    }

    /// Records that the value at `row` changed at `tick`.
    pub(crate) fn mark_changed(&mut self, row: usize, tick: Tick) {
        self.ticks[row].changed = tick;
    }

    /// Records that every value changed at `tick`.
    pub(crate) fn mark_all_changed(&mut self, tick: Tick) {
        self.bulk_changed = Some(tick);
    }

    /// Clamps every tick to at most [`MAX_CHANGE_AGE`](crate::tick::MAX_CHANGE_AGE) before `now`.
    pub(crate) fn clamp_ticks(&mut self, now: Tick) {
        for ticks in &mut self.ticks {
            ticks.added.clamp_age(now);
            ticks.changed.clamp_age(now);
        }
        if let Some(bulk) = &mut self.bulk_changed {
            bulk.clamp_age(now);
        }
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        self.clear();
        if let Some(ptr) = self.ptr {
            let item_layout = self.item_layout();
            if item_layout.size() > 0 && self.capacity > 0 {
                // SAFETY: ptr was allocated with alloc/realloc using this layout and capacity.
                unsafe {
                    let layout = Layout::from_size_align_unchecked(
                        item_layout.size() * self.capacity,
                        item_layout.align(),
                    );
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        }
    }
}
