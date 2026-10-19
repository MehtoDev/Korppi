//! Runtime-defined components (ADR-048): components without a Rust type, registered by name and
//! layout and accessed as bytes, for the editor's reflection-based inspector and scripting.
//!
//! A runtime component's values are only ever written as bytes, so all of their bytes are
//! initialized and reading them is sound. Typed components are refused by the byte API, since a
//! Rust value's padding bytes may be uninitialized, and typed access (`get_by_id::<T>`) refuses
//! runtime components, which have no `TypeId`.
//!
//! A runtime component without a drop function accepts any bytes. One with a drop function is
//! registered and inserted through `unsafe` functions, since the drop function may rely on the
//! bytes forming a valid value; its bytes cannot be written in place.

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::observer::EventKind;
use crate::world::World;
use std::alloc::Layout;

impl World {
    /// Registers a component without a Rust type whose values are plain bytes (any bit pattern
    /// is valid, nothing to drop). Registering a name again returns its component.
    ///
    /// # Panics
    /// If `name` is already registered with another layout or with a drop function.
    pub fn register_dynamic_component(&mut self, name: &str, layout: Layout) -> ComponentId {
        self.registry_mut().register_dynamic(name, layout, None)
    }

    /// Registers a component without a Rust type whose values `drop` must run on when they are
    /// removed, overwritten or despawned.
    ///
    /// # Safety
    /// `drop` must be sound to call once on each value inserted with
    /// [`insert_bytes_unchecked`](Self::insert_bytes_unchecked).
    ///
    /// # Panics
    /// If `name` is already registered with another layout or drop function.
    pub unsafe fn register_dynamic_component_with_drop(
        &mut self,
        name: &str,
        layout: Layout,
        drop: unsafe fn(*mut u8),
    ) -> ComponentId {
        self.registry_mut()
            .register_dynamic(name, layout, Some(drop))
    }

    /// The name of a component: its type name, or its runtime-defined name.
    #[must_use]
    pub fn component_name(&self, id: ComponentId) -> Option<&str> {
        self.components().name(id)
    }

    /// Adds the runtime component `id` to `entity` with the value `bytes`, or overwrites it.
    ///
    /// # Panics
    /// If `entity` is dead, `id` is not a runtime-defined component, has a drop function (use
    /// [`insert_bytes_unchecked`](Self::insert_bytes_unchecked)), or `bytes` is not its size.
    pub fn insert_bytes(&mut self, entity: Entity, id: ComponentId, bytes: &[u8]) {
        self.check_dynamic(id, bytes);
        let info = self.components().get_info(id).unwrap();
        assert!(
            !info.needs_drop(),
            "runtime-defined component {id:?} has a drop function; insert it with insert_bytes_unchecked"
        );
        // SAFETY: the component has no drop function, so any bytes of its size are a valid value.
        unsafe { self.insert_bytes_raw(entity, id, bytes) };
    }

    /// Adds the runtime component `id` to `entity` with the value `bytes`, or overwrites it
    /// (dropping the old value).
    ///
    /// # Safety
    /// `bytes` must form a valid value of the component for its drop function.
    ///
    /// # Panics
    /// As for [`insert_bytes`](Self::insert_bytes), except for the drop function.
    pub unsafe fn insert_bytes_unchecked(&mut self, entity: Entity, id: ComponentId, bytes: &[u8]) {
        self.check_dynamic(id, bytes);
        // SAFETY: the caller guarantees a valid value.
        unsafe { self.insert_bytes_raw(entity, id, bytes) };
    }

    /// The bytes of the runtime component `id` on `entity` (or inherited from a prefab), or `None`
    /// if the entity lacks it or `id` is not a runtime-defined component.
    #[must_use]
    pub fn get_bytes(&self, entity: Entity, id: ComponentId) -> Option<&[u8]> {
        if !self.components().is_dynamic(id) {
            return None;
        }
        let location = self.entity_location(entity)?;
        let arch = &self.archetypes()[location.archetype_id];
        if let Some(column) = arch.column(id) {
            // SAFETY: a runtime component's values are written only as bytes, so all are initialized.
            return Some(unsafe { column.row_bytes(location.row) });
        }
        let owner = arch.inherits(id)?;
        let owner_loc = self.entity_location(owner)?;
        let column = self.archetypes()[owner_loc.archetype_id].column(id)?;
        // SAFETY: as above, for the prefab owner's row.
        Some(unsafe { column.row_bytes(owner_loc.row) })
    }

    /// The bytes of the runtime component `id` on `entity`, mutably; `None` if the entity lacks
    /// it, `id` is not a runtime-defined component, or it has a drop function. Marks the
    /// component changed.
    pub fn get_bytes_mut(&mut self, entity: Entity, id: ComponentId) -> Option<&mut [u8]> {
        let info = self.components().get_info(id)?;
        if !self.components().is_dynamic(id) || info.needs_drop() {
            return None;
        }
        let location = self.entity_location(entity)?;
        let tick = self.change_tick();
        let column = self.archetypes_mut()[location.archetype_id].column_mut(id)?;
        column.mark_changed(location.row, tick);
        // SAFETY: initialized as in `get_bytes`; without a drop function any bytes are valid.
        Some(unsafe { column.row_bytes_mut(location.row) })
    }

    /// Removes the component `id` from `entity`, dropping its value; returns `false` if the
    /// entity lacks it or `id` is a relationship pair (removed with the pair API).
    pub fn remove_by_id(&mut self, entity: Entity, id: ComponentId) -> bool {
        if self.components().id_to_pair(id).is_some() {
            return false;
        }
        let removed = self.remove_component_by_id_raw(entity, id);
        if removed {
            self.observers_mut().record(EventKind::Remove, id, entity);
            self.flush_observers();
        }
        removed
    }

    /// The live entities that own or inherit component `id`.
    pub fn entities_with(&self, id: ComponentId) -> impl Iterator<Item = Entity> + '_ {
        self.archetypes()
            .iter()
            .filter(move |archetype| archetype.has_or_inherits(id))
            .flat_map(|archetype| archetype.entities().iter().copied())
    }

    /// The components of `entity` (including relationship pairs), in ID order.
    #[must_use]
    pub fn entity_components(&self, entity: Entity) -> Option<&[ComponentId]> {
        let location = self.entity_location(entity)?;
        Some(self.archetypes()[location.archetype_id].components())
    }

    fn check_dynamic(&self, id: ComponentId, bytes: &[u8]) {
        let registry = self.components();
        assert!(
            registry.is_dynamic(id),
            "{id:?} is not a runtime-defined component; typed components are accessed by type"
        );
        let size = registry.get_info(id).unwrap().layout().size();
        assert_eq!(
            bytes.len(),
            size,
            "runtime-defined component `{}` takes {size} bytes",
            registry.name(id).unwrap_or("?")
        );
    }

    /// Inserts or overwrites the runtime component `id` with `bytes`.
    ///
    /// # Safety
    /// `id` is a runtime component, `bytes` has its size and forms a valid value.
    unsafe fn insert_bytes_raw(&mut self, entity: Entity, id: ComponentId, bytes: &[u8]) {
        let location = self.entity_location(entity).expect("Entity not found");
        let tick = self.change_tick();
        if let Some(column) = self.archetypes_mut()[location.archetype_id].column_mut(id) {
            // SAFETY: the caller guarantees a valid value; the entity's row holds the old one.
            unsafe { column.replace_bytes(location.row, bytes, tick) };
            self.observers_mut().record(EventKind::Set, id, entity);
            self.flush_observers();
            return;
        }
        let dst = self.archetype_with_added(location, id);
        // SAFETY: as above, for the new row the move appends.
        unsafe {
            self.move_with_added(entity, location, id, dst, |column| {
                column.push_bytes(bytes, tick);
            });
        }
        self.observers_mut().record(EventKind::Add, id, entity);
        self.observers_mut().record(EventKind::Set, id, entity);
        self.flush_observers();
    }
}
