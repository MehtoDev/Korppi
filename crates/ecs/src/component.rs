//! Component trait, metadata, and type registry.

use std::alloc::Layout;
use std::any::TypeId;

/// Trait implemented by all data types that can be stored as ECS components.
pub trait Component: 'static + Send + Sync {}

/// Unique numeric identifier assigned to each registered component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ComponentId(pub(crate) usize);

impl ComponentId {
    /// Returns the raw numeric index of the component ID.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Metadata describing memory layout and cleanup functions for a component type.
#[derive(Debug, Clone, Copy)]
pub struct ComponentInfo {
    id: ComponentId,
    name: &'static str,
    layout: Layout,
    drop_fn: Option<unsafe fn(*mut u8)>,
    type_id: Option<TypeId>,
}

impl ComponentInfo {
    /// Returns the component's unique ID.
    #[must_use]
    pub const fn id(&self) -> ComponentId {
        self.id
    }

    /// Returns the human-readable type name (`"<runtime-defined>"` for a component registered at
    /// run time; [`ComponentRegistry::name`] has its name).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the memory layout (size and alignment) of the component.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns the Rust [`TypeId`] of the component type, if registered from a static type.
    #[must_use]
    pub const fn type_id(&self) -> Option<TypeId> {
        self.type_id
    }

    /// Returns `true` if this component requires a custom destructor.
    #[must_use]
    pub const fn needs_drop(&self) -> bool {
        self.drop_fn.is_some()
    }

    /// Safely drops a component instance pointed to by `ptr`.
    ///
    /// A `Drop` implementation that panics here aborts the process: the world drops components in
    /// the middle of structural changes, where unwinding would leave its tables inconsistent. The
    /// engine's release profile aborts on every panic anyway.
    ///
    /// # Safety
    /// `ptr` must point to a valid, initialized instance of the component described by this metadata.
    pub unsafe fn drop_in_place(&self, ptr: *mut u8) {
        if let Some(drop) = self.drop_fn {
            let abort_on_unwind = AbortOnUnwind;
            // SAFETY: caller guarantees `ptr` points to a valid instance described by this metadata.
            unsafe {
                drop(ptr);
            }
            std::mem::forget(abort_on_unwind);
        }
    }
}

use crate::entity::Entity;
use crate::fast_hash::FastHashMap;

/// Aborts the process when dropped, which only happens while unwinding out of a component's
/// `Drop`: the normal path forgets it.
struct AbortOnUnwind;

impl Drop for AbortOnUnwind {
    fn drop(&mut self) {
        std::process::abort();
    }
}

/// Central registry mapping Rust types and relationship pairs to unique [`ComponentId`]s.
#[derive(Default)]
pub struct ComponentRegistry {
    components: Vec<ComponentInfo>,
    type_to_id: FastHashMap<TypeId, ComponentId>,
    pair_to_id: FastHashMap<(ComponentId, Entity), ComponentId>,
    id_to_pair: FastHashMap<ComponentId, (ComponentId, Entity)>,
    target_pairs: FastHashMap<Entity, Vec<(ComponentId, ComponentId)>>,
    free_pair_ids: Vec<ComponentId>,
    /// Emptied `target_pairs` lists of dead targets, reused for new targets.
    free_pair_lists: Vec<Vec<(ComponentId, ComponentId)>>,
    /// Runtime-defined components by name, and each one's name.
    dynamic_by_name: FastHashMap<Box<str>, ComponentId>,
    dynamic_names: FastHashMap<ComponentId, Box<str>>,
}

impl ComponentRegistry {
    /// Creates an empty component registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a component type `T` and returns its unique [`ComponentId`].
    pub fn register<T: Component>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.type_to_id.get(&type_id) {
            return id;
        }

        let id = ComponentId(self.components.len());
        let info = ComponentInfo {
            id,
            name: std::any::type_name::<T>(),
            layout: Layout::new::<T>(),
            drop_fn: if std::mem::needs_drop::<T>() {
                Some(|ptr| {
                    // SAFETY: ptr is cast to *mut T and dropped in place as guaranteed by ComponentInfo::drop_in_place contract.
                    unsafe {
                        std::ptr::drop_in_place(ptr as *mut T);
                    }
                })
            } else {
                None
            },
            type_id: Some(type_id),
        };

        self.components.push(info);
        self.type_to_id.insert(type_id, id);
        id
    }

    /// Registers a component without a Rust type: `name` and `layout`, and `drop` to run on a
    /// value when it is removed. Registering a name again returns its component.
    ///
    /// # Panics
    /// If `name` is registered with another layout or drop function.
    pub(crate) fn register_dynamic(
        &mut self,
        name: &str,
        layout: Layout,
        drop_fn: Option<unsafe fn(*mut u8)>,
    ) -> ComponentId {
        if let Some(&id) = self.dynamic_by_name.get(name) {
            let info = &self.components[id.0];
            assert!(
                info.layout == layout && info.drop_fn.map(|f| f as usize) == drop_fn.map(|f| f as usize),
                "runtime-defined component `{name}` is already registered with another layout or drop function"
            );
            return id;
        }
        let id = ComponentId(self.components.len());
        self.components.push(ComponentInfo {
            id,
            name: "<runtime-defined>",
            layout,
            drop_fn,
            type_id: None,
        });
        self.dynamic_by_name.insert(name.into(), id);
        self.dynamic_names.insert(id, name.into());
        id
    }

    /// Returns `true` for a component registered at run time, without a Rust type.
    #[must_use]
    pub fn is_dynamic(&self, id: ComponentId) -> bool {
        self.dynamic_names.contains_key(&id)
    }

    /// The name of a component: its type name, or the name it was registered with at run time.
    #[must_use]
    pub fn name(&self, id: ComponentId) -> Option<&str> {
        match self.dynamic_names.get(&id) {
            Some(name) => Some(name),
            None => self.get_info(id).map(ComponentInfo::name),
        }
    }

    /// Registers a relationship pair `(relation_id, target)` and returns its unique [`ComponentId`].
    pub(crate) fn register_pair(
        &mut self,
        relation_id: ComponentId,
        target: Entity,
    ) -> ComponentId {
        if let Some(&id) = self.pair_to_id.get(&(relation_id, target)) {
            return id;
        }

        let rel_info = *self.get_info(relation_id).expect("Relation not registered");
        let id = if let Some(recycled_id) = self.free_pair_ids.pop() {
            recycled_id
        } else {
            ComponentId(self.components.len())
        };

        let info = ComponentInfo {
            id,
            name: "Pair",
            layout: rel_info.layout(),
            drop_fn: rel_info.drop_fn,
            type_id: rel_info.type_id,
        };

        if id.0 < self.components.len() {
            self.components[id.0] = info;
        } else {
            self.components.push(info);
        }

        self.pair_to_id.insert((relation_id, target), id);
        self.id_to_pair.insert(id, (relation_id, target));
        self.target_pairs
            .entry(target)
            .or_insert_with(|| self.free_pair_lists.pop().unwrap_or_default())
            .push((relation_id, id));
        id
    }

    /// Populates `out` with all unregistered relationship pairs targeting `target`, recycling their component IDs.
    pub(crate) fn unregister_all_pairs_for_target_into(
        &mut self,
        target: Entity,
        out: &mut Vec<ComponentId>,
    ) {
        out.clear();
        let Some(mut pairs) = self.target_pairs.remove(&target) else {
            return;
        };

        for &(relation_id, pair_id) in &pairs {
            self.pair_to_id.remove(&(relation_id, target));
            self.id_to_pair.remove(&pair_id);
            self.free_pair_ids.push(pair_id);
            out.push(pair_id);
        }
        pairs.clear();
        self.free_pair_lists.push(pairs);
    }

    /// Returns an iterator over all pair component IDs targeting `target`.
    pub fn pairs_for_target(&self, target: Entity) -> impl Iterator<Item = ComponentId> + '_ {
        self.target_pairs
            .get(&target)
            .into_iter()
            .flat_map(|pairs| pairs.iter().map(|&(_, pair_id)| pair_id))
    }

    /// Retrieves the component ID for type `T` if already registered.
    #[must_use]
    pub fn id_of<T: Component>(&self) -> Option<ComponentId> {
        self.type_to_id.get(&TypeId::of::<T>()).copied()
    }

    /// Retrieves the component ID for the pair `(relation_id, target)` if already registered.
    #[must_use]
    pub fn pair_to_id(&self, relation_id: ComponentId, target: Entity) -> Option<ComponentId> {
        self.pair_to_id.get(&(relation_id, target)).copied()
    }

    /// Retrieves the `(relation_id, target)` tuple for a registered pair component ID.
    #[must_use]
    pub fn id_to_pair(&self, id: ComponentId) -> Option<(ComponentId, Entity)> {
        self.id_to_pair.get(&id).copied()
    }

    /// Retrieves the metadata for the given component ID.
    #[must_use]
    pub fn get_info(&self, id: ComponentId) -> Option<&ComponentInfo> {
        self.components.get(id.0)
    }
}
