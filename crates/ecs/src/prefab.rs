//! Prefabs with Flecs' shared model (ADR-048).
//!
//! A prefab is an entity with the [`Prefab`] tag; queries skip archetypes holding the tag unless a
//! term names it. An instance holds the pair `(IsA, prefab)` and inherits the prefab's components
//! (except `Prefab` and relationship pairs): `get`, `has`, `&T`, `Option<&T>`, `Has<T>`, `With<T>`
//! and `Without<T>` look at the instance first and then at its prefabs, transitively, in pair
//! order. The instance overrides a component by adding its own value; `&mut T` terms, `get_mut`
//! and the change filters see only owned components, so an instance never writes its prefab.
//!
//! Since an `IsA` pair is an archetype component, all entities of an archetype share their
//! prefabs. Each archetype keeps the list of components it inherits and the prefab entity owning
//! each value, rebuilt when the archetype is created and whenever one of its prefabs changes
//! archetype, so queries resolve an inherited term once per archetype.

use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::relation::{ChildOf, IsA};
use crate::world::World;

/// Marks an entity as a prefab: a template whose components instances inherit through `IsA`.
/// Queries skip prefabs unless a term names `Prefab`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Prefab;
impl Component for Prefab {}

impl World {
    /// Creates an instance of `prefab`: a new entity with the pair `(IsA, prefab)`, and an
    /// instance of each of the prefab's `ChildOf` children, attached to it in the same order
    /// (recursively). Returns the instance of `prefab`.
    ///
    /// # Panics
    /// If `prefab` is not alive.
    pub fn instantiate(&mut self, prefab: Entity) -> Entity {
        assert!(self.is_alive(prefab), "{prefab:?} is not alive");
        let mut queue = std::mem::take(&mut self.instantiate_queue);
        queue.clear();
        queue.push((prefab, None));
        let mut root = None;
        let mut next = 0;
        while let Some(&(template, parent)) = queue.get(next) {
            next += 1;
            let instance = self.spawn_empty();
            self.add_pair::<IsA>(instance, template);
            match parent {
                Some(parent) => self.add_pair::<ChildOf>(instance, parent),
                None => root = Some(instance),
            }
            queue.extend(
                self.children(template)
                    .iter()
                    .map(|&child| (child, Some(instance))),
            );
        }
        self.instantiate_queue = queue;
        root.expect("the prefab itself is instantiated first")
    }

    /// Rebuilds the list of components archetype `archetype` inherits through its `IsA` pairs.
    pub(crate) fn refresh_inherited(&mut self, archetype: usize) {
        let mut inherited = self.archetypes_mut()[archetype].take_inherited();
        inherited.clear();
        let Some(isa) = self.isa_id() else {
            self.archetypes_mut()[archetype].set_inherited(inherited);
            return;
        };
        let prefab = self.prefab_id();
        let arch = &self.archetypes()[archetype];
        for &(relation, target) in arch.pairs() {
            if relation != isa {
                continue;
            }
            let Some(location) = self.entity_location(target) else {
                continue;
            };
            let source = &self.archetypes()[location.archetype_id];
            let owned = source
                .components()
                .iter()
                .filter(|&&c| Some(c) != prefab && self.components().id_to_pair(c).is_none())
                .map(|&c| (c, target));
            let transitive = source.inherited().iter().copied();
            for (component, owner) in owned.chain(transitive) {
                if !arch.has_component(component) && inherited.iter().all(|&(c, _)| c != component)
                {
                    inherited.push((component, owner));
                }
            }
        }
        inherited.sort_unstable_by_key(|&(c, _)| c);
        self.archetypes_mut()[archetype].set_inherited(inherited);
    }

    /// After `prefab` changed archetype: rebuilds the inherited lists of its instances'
    /// archetypes, and of their instances in turn.
    pub(crate) fn refresh_instances_of(&mut self, prefab: Entity) {
        let Some(pair) = self
            .isa_id()
            .and_then(|isa| self.components().pair_to_id(isa, prefab))
        else {
            return;
        };
        let mut i = 0;
        while let Some(&archetype) = self.pair_archetypes_of(pair).get(i) {
            i += 1;
            self.refresh_inherited(archetype);
            let mut row = 0;
            while let Some(&entity) = self.archetypes()[archetype].entities().get(row) {
                row += 1;
                if self.entity_manager().is_isa_target(entity) {
                    self.refresh_instances_of(entity);
                }
            }
        }
        self.bump_archetype_epoch();
    }

    /// The inherited value of `T` for an entity in archetype `archetype`, if it inherits one.
    pub(crate) fn inherited_component<T: 'static>(
        &self,
        archetype: usize,
        id: ComponentId,
    ) -> Option<&T> {
        let owner = self.archetypes()[archetype].inherits(id)?;
        let location = self.entity_location(owner)?;
        let column = self.archetypes()[location.archetype_id].column(id)?;
        if column.info().type_id() != Some(std::any::TypeId::of::<T>()) {
            return None;
        }
        // SAFETY: the column stores `T` (checked above), and the owner's row holds a live value.
        Some(unsafe { column.get::<T>(location.row) })
    }
}
