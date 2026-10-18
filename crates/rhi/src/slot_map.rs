//! Generational slot maps for resource registries (findings RHI-10, RHI-15).
//!
//! Handles carry an index, a generation and a device. A [`SlotMap`] stores values in a `Vec` of
//! slots with a free list: inserting reuses a freed slot, removing bumps the slot's generation,
//! and every lookup by handle compares generations, so a handle to a destroyed resource never
//! reaches the resource that later reuses its slot. A map belongs to one device
//! ([`SlotMap::for_device`]) and finds no key of another device, whose slots are numbered
//! independently. Lookups are O(1) without hashing.
//!
//! Values are stored as `(value, generation)` pairs, the shape the registries had as
//! `HashMap`s, so call sites destructure them the same way.

use crate::handle::DeviceId;

/// The index, generation and device a lookup checks.
pub trait SlotKey: Copy {
    /// Slot index.
    fn slot(&self) -> u32;
    /// Generation the slot must have.
    fn slot_generation(&self) -> u32;
    /// Device the map must belong to.
    fn slot_device(&self) -> DeviceId;
}

impl<K: SlotKey> SlotKey for &K {
    fn slot(&self) -> u32 {
        (*self).slot()
    }

    fn slot_generation(&self) -> u32 {
        (*self).slot_generation()
    }

    fn slot_device(&self) -> DeviceId {
        (*self).slot_device()
    }
}

/// A handle type a [`SlotMap`] can mint ([`SlotMap::insert_handle`]).
pub trait SlotHandle: SlotKey {
    /// The handle of `device`'s slot `id` at `generation`.
    fn from_slot(device: DeviceId, id: u32, generation: u32) -> Self;
}

struct Slot<T> {
    generation: u32,
    value: Option<(T, u32)>,
}

/// A generational slot map (see the module documentation).
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
    device: DeviceId,
}

impl<T> Default for SlotMap<T> {
    /// A map of no device ([`DeviceId::NONE`]).
    fn default() -> Self {
        Self::for_device(DeviceId::NONE)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for SlotMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<T> SlotMap<T> {
    /// An empty map of no device ([`DeviceId::NONE`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty map of `device`'s resources.
    pub fn for_device(device: DeviceId) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            device,
        }
    }

    /// The device whose resources the map holds.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Stores `value` and returns its handle, of this map's device.
    pub fn insert_handle<H: SlotHandle>(&mut self, value: T) -> H {
        let (id, generation) = self.insert(value);
        H::from_slot(self.device, id, generation)
    }

    /// Stores `value` and returns its `(index, generation)`; generations start at 1.
    pub fn insert(&mut self, value: T) -> (u32, u32) {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some((value, slot.generation));
            (index, slot.generation)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                value: Some((value, 1)),
            });
            (index, 1)
        }
    }

    fn slot(&self, key: impl SlotKey) -> Option<&Slot<T>> {
        if key.slot_device() != self.device {
            return None;
        }
        self.slots
            .get(key.slot() as usize)
            .filter(|slot| slot.generation == key.slot_generation())
    }

    /// The value of a live `key`.
    pub fn get(&self, key: impl SlotKey) -> Option<&(T, u32)> {
        self.slot(key).and_then(|slot| slot.value.as_ref())
    }

    /// The value of a live `key`, mutably.
    pub fn get_mut(&mut self, key: impl SlotKey) -> Option<&mut (T, u32)> {
        if key.slot_device() != self.device {
            return None;
        }
        self.slots
            .get_mut(key.slot() as usize)
            .filter(|slot| slot.generation == key.slot_generation())
            .and_then(|slot| slot.value.as_mut())
    }

    /// True if `key` is live.
    pub fn contains(&self, key: impl SlotKey) -> bool {
        self.get(key).is_some()
    }

    /// The value in slot `index`, whatever its generation. Only for indices the caller knows to
    /// be live (internal caches invalidated when their entries are removed).
    pub fn get_by_index(&self, index: u32) -> Option<&(T, u32)> {
        self.slots
            .get(index as usize)
            .and_then(|slot| slot.value.as_ref())
    }

    /// The value in slot `index` mutably, whatever its generation (see
    /// [`SlotMap::get_by_index`]).
    pub fn get_by_index_mut(&mut self, index: u32) -> Option<&mut (T, u32)> {
        self.slots
            .get_mut(index as usize)
            .and_then(|slot| slot.value.as_mut())
    }

    /// Removes a live `key`'s value and makes the slot's next value a new generation.
    pub fn remove(&mut self, key: impl SlotKey) -> Option<(T, u32)> {
        if key.slot_device() != self.device {
            return None;
        }
        let index = key.slot();
        let slot = self
            .slots
            .get_mut(index as usize)
            .filter(|slot| slot.generation == key.slot_generation() && slot.value.is_some())?;
        let value = slot.value.take();
        // Generation 0 is never handed out, so null-like handles never match.
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free.push(index);
        self.len -= 1;
        value
    }

    /// Number of live values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no value is live.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Live `(index, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &(T, u32))> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.value.as_ref().map(|v| (i as u32, v)))
    }

    /// Live values.
    pub fn values(&self) -> impl Iterator<Item = &(T, u32)> {
        self.slots.iter().filter_map(|slot| slot.value.as_ref())
    }

    /// Live values, mutably.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut (T, u32)> {
        self.slots.iter_mut().filter_map(|slot| slot.value.as_mut())
    }

    /// Removes and returns every value (teardown); the map stays usable and its slots are
    /// reused with new generations.
    pub fn drain(&mut self) -> Vec<(u32, (T, u32))> {
        let mut drained = Vec::with_capacity(self.len);
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let Some(value) = slot.value.take() {
                slot.generation = slot.generation.wrapping_add(1).max(1);
                drained.push((index as u32, value));
            }
        }
        self.len = 0;
        self.free = (0..self.slots.len() as u32).rev().collect();
        drained
    }

    /// Removes every value.
    pub fn clear(&mut self) {
        self.drain();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::BufferHandle;

    #[test]
    fn keys_of_another_device_miss_even_in_the_same_slot() {
        let (a, b) = (DeviceId::next(), DeviceId::next());
        assert_ne!(a, b);
        let mut map_a = SlotMap::for_device(a);
        let mut map_b = SlotMap::for_device(b);
        let in_a: BufferHandle = map_a.insert_handle("a");
        let in_b: BufferHandle = map_b.insert_handle("b");
        assert_eq!(
            (in_a.id(), in_a.generation()),
            (in_b.id(), in_b.generation())
        );
        assert_eq!(in_a.device(), a);
        assert!(map_b.get(in_a).is_none());
        assert!(map_b.get_mut(in_a).is_none());
        assert!(map_b.remove(in_a).is_none());
        assert_eq!(
            map_b.get(in_b).map(|(v, _)| *v),
            Some("b"),
            "b's value is untouched"
        );
        // A handle of no device misses a device's map, and a device's handle misses a map of no
        // device.
        let unowned = BufferHandle::new(in_a.id(), in_a.generation());
        assert!(map_a.get(unowned).is_none());
        assert!(SlotMap::<&str>::new().get(in_a).is_none());
    }

    #[test]
    fn stale_handles_miss_reused_slots() {
        let mut map = SlotMap::new();
        let (id, generation) = map.insert("first");
        let first = BufferHandle::new(id, generation);
        assert_eq!(map.get(first).map(|(v, _)| *v), Some("first"));
        assert_eq!(map.remove(first).map(|(v, _)| v), Some("first"));
        assert!(map.remove(first).is_none(), "removed once");
        let (id2, generation2) = map.insert("second");
        assert_eq!(id2, id, "the slot is reused");
        assert_ne!(generation2, generation);
        assert!(map.get(first).is_none(), "the stale handle misses");
        let second = BufferHandle::new(id2, generation2);
        assert_eq!(map.get(second).map(|(v, _)| *v), Some("second"));
        assert_eq!(map.len(), 1);
        assert!(map.get(BufferHandle::null()).is_none());
    }

    #[test]
    fn drain_and_clear_empty_the_map() {
        let mut map = SlotMap::new();
        let keys: Vec<_> = (0..4).map(|i| map.insert(i)).collect();
        assert_eq!(map.drain().len(), 4);
        assert!(map.is_empty());
        for (id, generation) in keys {
            assert!(map.get(BufferHandle::new(id, generation)).is_none());
        }
        map.insert(9);
        map.clear();
        assert!(map.is_empty());
        let (_, generation) = map.insert(1);
        assert!(generation > 1);
    }
}
