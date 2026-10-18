//! High-performance archetype-based Entity Component System (ECS).
//!
//! # Soundness
//!
//! The `unsafe` code in the query iterator and the column storage relies on invariants that only
//! [`World`] maintains: every archetype's columns and entity list have one row per entity, a
//! column stores the type its component ID was registered with, and a pair ID is in use by no
//! archetype once it is recycled. Safe code outside the crate therefore cannot build or change
//! archetypes, iterate archetypes it assembled itself, free component IDs, or implement
//! [`WorldQuery`] without `unsafe`. Each of these fails to compile:
//!
//! ```compile_fail
//! // A query iterator over hand-assembled archetypes.
//! let mut archetypes: Vec<korppi_ecs::archetype::Archetype> = Vec::new();
//! let _ = korppi_ecs::QueryIter::<korppi_ecs::Entity>::new((), &mut archetypes);
//! ```
//!
//! ```compile_fail
//! // An archetype assembled outside the world, holding an entity but no column values.
//! use korppi_ecs::archetype::Archetype;
//! let mut archetype = Archetype::with_pairs(0, Vec::new(), Vec::new(), Vec::new());
//! archetype.push_entity(korppi_ecs::World::new().spawn(()));
//! ```
//!
//! ```compile_fail
//! // Pair IDs freed while an archetype still uses them.
//! let mut world = korppi_ecs::World::new();
//! let target = world.spawn(());
//! world.components_mut().unregister_all_pairs_for_target(target);
//! ```
//!
//! ```compile_fail
//! // A query term whose reported accesses the iterator would have to trust.
//! use korppi_ecs::archetype::Archetype;
//! use korppi_ecs::component::ComponentRegistry;
//! use korppi_ecs::query::ComponentAccess;
//! struct Nothing;
//! impl korppi_ecs::WorldQuery for Nothing {
//!     type Item<'w> = ();
//!     type Slice<'w> = ();
//!     type State = ();
//!     type Fetch<'w> = ();
//!     fn empty_fetch<'w>() -> Self::Fetch<'w> {}
//!     fn init_state(_: &mut ComponentRegistry) -> Self::State {}
//!     fn access_set(_: &Self::State, _: &mut dyn FnMut(ComponentAccess)) {}
//!     fn matches_archetype(_: &Self::State, _: &Archetype) -> bool { true }
//!     unsafe fn init_fetch<'w>(
//!         _: &Self::State,
//!         _: *mut Archetype,
//!         _: korppi_ecs::tick::ChangeTicks,
//!     ) -> Self::Fetch<'w> {
//!     }
//!     unsafe fn fetch_next<'w>(_: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
//!         ((), ())
//!     }
//!     unsafe fn fetch_slice<'w>(_: Self::Fetch<'w>, _: usize) -> Self::Slice<'w> {}
//! }
//! ```

pub mod archetype;
pub mod bundle;
pub mod command;
pub mod component;
mod dynamic;
pub mod entity;
pub(crate) mod fast_hash;
pub mod observer;
pub mod prefab;
pub mod query;
pub mod relation;
pub mod schedule;
pub mod storage;
pub mod system;
pub mod tick;
pub mod world;

pub use bundle::Bundle;
pub use command::{Command, Commands};
pub use component::{Component, ComponentId};
pub use entity::Entity;
pub use observer::{ObserverId, OnAdd, OnRemove, OnSet};
pub use prefab::Prefab;
pub use query::{Added, Changed, Has, QueryIter, QueryState, Target, With, Without, WorldQuery};
pub use relation::{
    ChildOf, CleanupPolicy, IsA, Relation, RelationshipError, WithPair, WithRelation,
    WithoutRelation,
};
pub use schedule::{Schedule, Stage, SystemConfig};
pub use system::{IntoSystem, Local, Query, Res, ResMut, System, SystemParam};
pub use world::{AncestorIter, DescendantIter, World};

/// Common ECS types re-exported for convenience.
pub mod prelude {
    pub use crate::bundle::Bundle;
    pub use crate::command::{Command, Commands};
    pub use crate::component::{Component, ComponentId};
    pub use crate::entity::Entity;
    pub use crate::observer::{ObserverId, OnAdd, OnRemove, OnSet};
    pub use crate::prefab::Prefab;
    pub use crate::query::{
        Added, Changed, Has, QueryIter, QueryState, Target, With, Without, WorldQuery,
    };
    pub use crate::relation::{
        ChildOf, CleanupPolicy, IsA, Relation, RelationshipError, WithPair, WithRelation,
        WithoutRelation,
    };
    pub use crate::schedule::{Schedule, Stage, SystemConfig};
    pub use crate::system::{IntoSystem, Local, Query, Res, ResMut, System, SystemParam};
    pub use crate::world::{AncestorIter, DescendantIter, World};
}
