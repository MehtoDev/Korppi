//! Systems: functions whose parameters say what they access (ADR-048).
//!
//! A system is a function or closure whose parameters implement [`SystemParam`]: [`Query`],
//! [`Res`], [`ResMut`], [`Commands`], [`Local`], or a tuple of them. The schedule builds each
//! parameter's state on the system's first run, checks that no two parameters access the same
//! component or resource with at least one of them writing, and hands the function its
//! parameters on every run. Commands are applied when the function returns. A function that takes
//! `&mut World` alone is an exclusive system.
//!
//! Change detection is per system: [`Added`](crate::Added) and [`Changed`](crate::Changed) terms
//! in a system's queries compare with the tick of that system's previous run.

use crate::archetype::Archetype;
use crate::command::{CommandQueue, Commands};
use crate::entity::Entity;
use crate::query::{ComponentAccess, FetchContext, QueryIter, QueryState, WorldQuery};
use crate::tick::{ChangeTicks, Tick, MAX_CHANGE_AGE};
use crate::world::World;
use std::any::{type_name, TypeId};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// The component and resource accesses of one system's parameters.
#[derive(Debug, Default)]
pub struct SystemAccess {
    components: Vec<ComponentAccess>,
    /// `(type, name, writes)` per resource parameter.
    resources: Vec<(TypeId, &'static str, bool)>,
}

impl SystemAccess {
    /// Records a component access; panics if it conflicts with one recorded earlier.
    pub fn add_component(&mut self, access: ComponentAccess, name: &str) {
        let (id, writes) = match access {
            ComponentAccess::Read(id) => (id, false),
            ComponentAccess::Write(id) => (id, true),
        };
        for &earlier in &self.components {
            let (other, other_writes) = match earlier {
                ComponentAccess::Read(other) => (other, false),
                ComponentAccess::Write(other) => (other, true),
            };
            assert!(
                other != id || !(writes || other_writes),
                "Conflicting access in system parameters: component `{name}` is borrowed mutably by one parameter and also by another"
            );
        }
        self.components.push(access);
    }

    /// Records a resource access; panics if it conflicts with one recorded earlier.
    pub fn add_resource(&mut self, id: TypeId, name: &'static str, writes: bool) {
        for &(other, _, other_writes) in &self.resources {
            assert!(
                other != id || !(writes || other_writes),
                "Conflicting access in system parameters: resource `{name}` is borrowed mutably by one parameter and also by another"
            );
        }
        self.resources.push((id, name, writes));
    }
}

/// The world during a system run, split into the parts parameters borrow.
///
/// Created once per run from `&mut World`, so that every parameter derives its pointers from
/// the same exclusive borrow; [`SystemAccess`] has checked that the parameters' borrows are
/// disjoint.
#[derive(Clone, Copy)]
pub struct WorldCell<'w> {
    world: *mut World,
    archetypes: *mut [Archetype],
    _marker: PhantomData<&'w mut World>,
}

impl<'w> WorldCell<'w> {
    fn new(world: &'w mut World) -> Self {
        let world: *mut World = world;
        // SAFETY: `world` comes from a live exclusive borrow; the archetype slice is taken once,
        // here, and shared by every query parameter of this run.
        let archetypes = unsafe { (*world).archetypes_mut() as *mut [Archetype] };
        Self {
            world,
            archetypes,
            _marker: PhantomData,
        }
    }

    /// The world, for reads that no parameter writes (entity locations, the registry).
    ///
    /// # Safety
    /// The caller may not read anything a parameter of this run writes.
    pub unsafe fn world(self) -> &'w World {
        // SAFETY: the caller guarantees the reads do not overlap parameter writes.
        unsafe { &*self.world }
    }

    /// The raw world pointer, for parameters that project their own disjoint part.
    #[must_use]
    pub fn as_ptr(self) -> *mut World {
        self.world
    }

    /// The archetype tables, shared by every query parameter of this run.
    #[must_use]
    pub fn archetypes(self) -> *mut [Archetype] {
        self.archetypes
    }
}

/// A value a system can take as a parameter.
///
/// # Safety
/// `init_state` must record in `access` everything `get_param` reads or writes, and `get_param`
/// must access nothing else: the schedule relies on the recorded accesses to keep the
/// parameters' borrows disjoint.
pub unsafe trait SystemParam {
    /// Built on the system's first run and kept between runs.
    type State: Send + 'static;
    /// The parameter the function receives.
    type Item<'w, 's>;

    /// Builds the parameter's state and records its accesses.
    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State;

    /// Brings the state up to date before a run (for example, new archetypes for a query).
    fn update_state(_state: &mut Self::State, _world: &World) {}

    /// Builds the parameter for one run.
    ///
    /// # Safety
    /// `world` must be the world `init_state` saw, split for this run, and no other parameter of
    /// the run may access what this one records.
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        world: WorldCell<'w>,
        ticks: ChangeTicks,
    ) -> Self::Item<'w, 's>;

    /// Called after the function returns, with the whole world (applies commands).
    fn apply(_state: &mut Self::State, _world: &mut World) {}
}

/// The item type a parameter `P` hands the function.
pub type SystemParamItem<'w, 's, P> = <P as SystemParam>::Item<'w, 's>;

/// A query parameter: iterates the entities that match `Q`, with change detection relative to
/// the system's previous run.
pub struct Query<'w, 's, Q: WorldQuery> {
    state: &'s QueryState<Q>,
    world: WorldCell<'w>,
    ticks: ChangeTicks,
}

impl<'w, 's, Q: WorldQuery> Query<'w, 's, Q> {
    /// Iterates the matching entities.
    pub fn iter_mut(&mut self) -> QueryIter<'_, Q> {
        // SAFETY: the system's access check keeps this query's accesses disjoint from the other
        // parameters', and `&mut self` keeps two iterators of this query from overlapping.
        unsafe {
            QueryIter::from_raw(
                *self.state.state(),
                self.world.archetypes(),
                self.world.world().entity_manager(),
                Some(self.state.matched_archetypes()),
                self.ticks,
                false,
            )
        }
    }

    /// Executes `f` on every matching entity.
    pub fn for_each(&mut self, f: impl FnMut(Q::Item<'_>)) {
        self.iter_mut().for_each(f);
    }

    /// Executes `f` once per matching archetype, with its rows as slices.
    pub fn for_each_chunk(&mut self, f: impl FnMut(Q::Slice<'_>)) {
        self.iter_mut().for_each_chunk(f);
    }

    /// Returns the query's item for `entity`, or `None` if the entity is dead or does not match.
    pub fn get_mut(&mut self, entity: Entity) -> Option<Q::Item<'_>> {
        // SAFETY: entity locations are never written during a system run.
        let location = unsafe { self.world.world() }.entity_location(entity)?;
        let matched = self.state.matched_archetypes();
        matched.binary_search(&location.archetype_id).ok()?;
        // SAFETY: as in `iter_mut`; the archetype index comes from a live entity's location.
        unsafe {
            let archetype = (self.world.archetypes() as *mut Archetype).add(location.archetype_id);
            let state = self.state.state();
            let entities = self.world.world().entity_manager();
            let ctx = FetchContext::new(self.world.archetypes(), entities, self.ticks);
            let fetch = Q::seek(Q::init_fetch(state, archetype, ctx), location.row);
            if !Q::DENSE && !Q::filter(fetch) {
                return None;
            }
            Q::mark_row(state, archetype, location.row, self.ticks);
            Some(Q::fetch_next(fetch).0)
        }
    }
}

// SAFETY: records the query's component accesses; reads only those columns (through the query
// iterator) and the entity locations, which no parameter writes.
unsafe impl<Q: WorldQuery + 'static> SystemParam for Query<'_, '_, Q> {
    type State = QueryState<Q>;
    type Item<'w, 's> = Query<'w, 's, Q>;

    fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
        let state = world.query_state::<Q>();
        Q::access_set(state.state(), &mut |a| {
            let id = match a {
                ComponentAccess::Read(id) | ComponentAccess::Write(id) => id,
            };
            let name = world
                .components()
                .get_info(id)
                .map_or("unknown", |info| info.name());
            access.add_component(a, name);
        });
        state
    }

    fn update_state(state: &mut Self::State, world: &World) {
        state.update(world);
    }

    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        world: WorldCell<'w>,
        ticks: ChangeTicks,
    ) -> Self::Item<'w, 's> {
        Query {
            state,
            world,
            ticks,
        }
    }
}

/// Shared access to a resource of type `T`. The system panics if the resource is missing.
pub struct Res<'w, T: Send + Sync + 'static> {
    value: &'w T,
}

impl<T: Send + Sync + 'static> Deref for Res<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

/// Exclusive access to a resource of type `T`. The system panics if the resource is missing.
pub struct ResMut<'w, T: Send + Sync + 'static> {
    value: &'w mut T,
}

impl<T: Send + Sync + 'static> Deref for ResMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: Send + Sync + 'static> DerefMut for ResMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

/// The resource `T` of the world behind `world`.
///
/// # Safety
/// No other live reference may point at that resource for `'w` if the result is used mutably,
/// nor a mutable one otherwise.
unsafe fn resource_ptr<'w, T: Send + Sync + 'static>(world: WorldCell<'w>) -> &'w mut T {
    // SAFETY: the caller guarantees no conflicting borrow of this resource; the resource map is
    // a separate field, not borrowed by any other parameter during the run.
    unsafe { World::resource_raw::<T>(world.as_ptr()) }
        .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()))
}

// SAFETY: records a read of resource `T` and reads only it.
unsafe impl<T: Send + Sync + 'static> SystemParam for Res<'_, T> {
    type State = ();
    type Item<'w, 's> = Res<'w, T>;

    fn init_state(_world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.add_resource(TypeId::of::<T>(), type_name::<T>(), false);
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        world: WorldCell<'w>,
        _ticks: ChangeTicks,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the access check rules out a `ResMut<T>` in the same system.
        let value = unsafe { resource_ptr::<T>(world) };
        Res { value }
    }
}

// SAFETY: records a write of resource `T` and accesses only it.
unsafe impl<T: Send + Sync + 'static> SystemParam for ResMut<'_, T> {
    type State = ();
    type Item<'w, 's> = ResMut<'w, T>;

    fn init_state(_world: &mut World, access: &mut SystemAccess) -> Self::State {
        access.add_resource(TypeId::of::<T>(), type_name::<T>(), true);
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        world: WorldCell<'w>,
        _ticks: ChangeTicks,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the access check rules out any other access to `T` in the same system.
        let value = unsafe { resource_ptr::<T>(world) };
        ResMut { value }
    }
}

// SAFETY: reads only the entity manager's reservation counter (atomic) and its free list, which
// no parameter writes during a run; the queue is the parameter's own state.
unsafe impl SystemParam for Commands<'_, '_> {
    type State = CommandQueue;
    type Item<'w, 's> = Commands<'w, 's>;

    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {
        CommandQueue::new()
    }

    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        world: WorldCell<'w>,
        _ticks: ChangeTicks,
    ) -> Self::Item<'w, 's> {
        // SAFETY: entity records are never written during a system run.
        Commands::new(state, unsafe { world.world() }.entity_manager())
    }

    fn apply(state: &mut Self::State, world: &mut World) {
        state.apply(world);
    }
}

/// State owned by one system and kept between its runs, starting from `T::default()`.
pub struct Local<'s, T: Default + Send + 'static>(&'s mut T);

impl<T: Default + Send + 'static> Deref for Local<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0
    }
}

impl<T: Default + Send + 'static> DerefMut for Local<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0
    }
}

// SAFETY: accesses only its own state.
unsafe impl<T: Default + Send + 'static> SystemParam for Local<'_, T> {
    type State = T;
    type Item<'w, 's> = Local<'s, T>;

    fn init_state(_world: &mut World, _access: &mut SystemAccess) -> Self::State {
        T::default()
    }

    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        _world: WorldCell<'w>,
        _ticks: ChangeTicks,
    ) -> Self::Item<'w, 's> {
        Local(state)
    }
}

macro_rules! impl_param_tuple {
    ($($p:ident),*) => {
        // SAFETY: each element records its accesses in the same `SystemAccess`, which rejects
        // conflicts between them, and accesses only what it recorded.
        #[allow(non_snake_case, unused_variables, unused_unsafe, clippy::unused_unit)]
        unsafe impl<$($p: SystemParam),*> SystemParam for ($($p,)*) {
            type State = ($($p::State,)*);
            type Item<'w, 's> = ($($p::Item<'w, 's>,)*);

            fn init_state(world: &mut World, access: &mut SystemAccess) -> Self::State {
                ($($p::init_state(world, access),)*)
            }

            fn update_state(state: &mut Self::State, world: &World) {
                let ($($p,)*) = state;
                $($p::update_state($p, world);)*
            }

            unsafe fn get_param<'w, 's>(
                state: &'s mut Self::State,
                world: WorldCell<'w>,
                ticks: ChangeTicks,
            ) -> Self::Item<'w, 's> {
                let ($($p,)*) = state;
                // SAFETY: the caller's guarantee covers each element.
                unsafe { ($($p::get_param($p, world, ticks),)*) }
            }

            fn apply(state: &mut Self::State, world: &mut World) {
                let ($($p,)*) = state;
                $($p::apply($p, world);)*
            }
        }
    };
}

impl_param_tuple!();
impl_param_tuple!(P0);
impl_param_tuple!(P0, P1);
impl_param_tuple!(P0, P1, P2);
impl_param_tuple!(P0, P1, P2, P3);
impl_param_tuple!(P0, P1, P2, P3, P4);
impl_param_tuple!(P0, P1, P2, P3, P4, P5);
impl_param_tuple!(P0, P1, P2, P3, P4, P5, P6);
impl_param_tuple!(P0, P1, P2, P3, P4, P5, P6, P7);

/// A function usable as a system, with the parameters `Marker` names (`fn(P0, P1, ...)`).
pub trait SystemParamFunction<Marker>: Send + 'static {
    /// The function's parameters, as one tuple parameter.
    type Param: SystemParam;

    /// Calls the function.
    fn run(&mut self, param: SystemParamItem<'_, '_, Self::Param>);
}

macro_rules! impl_system_function {
    ($($p:ident),*) => {
        #[allow(non_snake_case)]
        impl<Func, $($p: SystemParam),*> SystemParamFunction<fn($($p,)*)> for Func
        where
            Func: Send + 'static,
            for<'a> &'a mut Func: FnMut($($p),*) + FnMut($(SystemParamItem<$p>),*),
        {
            type Param = ($($p,)*);

            fn run(&mut self, param: SystemParamItem<'_, '_, ($($p,)*)>) {
                // Calling through a generic function lets the compiler pick the impl of `FnMut`
                // for the parameters' item types.
                #[allow(clippy::too_many_arguments)]
                fn call_inner<$($p),*>(mut f: impl FnMut($($p),*), $($p: $p),*) {
                    f($($p),*);
                }
                let ($($p,)*) = param;
                call_inner(self, $($p),*);
            }
        }
    };
}

impl<Func> SystemParamFunction<fn()> for Func
where
    Func: FnMut() + Send + 'static,
{
    type Param = ();

    fn run(&mut self, _param: SystemParamItem<'_, '_, ()>) {
        self();
    }
}

impl_system_function!(P0);
impl_system_function!(P0, P1);
impl_system_function!(P0, P1, P2);
impl_system_function!(P0, P1, P2, P3);
impl_system_function!(P0, P1, P2, P3, P4);
impl_system_function!(P0, P1, P2, P3, P4, P5);
impl_system_function!(P0, P1, P2, P3, P4, P5, P6);
impl_system_function!(P0, P1, P2, P3, P4, P5, P6, P7);

/// A runnable system, as the schedule stores it.
pub trait System: Send + 'static {
    /// The system's default name: its function's type name.
    fn default_name(&self) -> &'static str;

    /// Runs the system once, applying its commands afterwards.
    fn run(&mut self, world: &mut World);
}

/// A [`SystemParamFunction`] with its parameters' state.
pub struct FunctionSystem<Marker, F: SystemParamFunction<Marker>> {
    func: F,
    /// The world the parameters were built for, and their state.
    state: Option<(u64, <F::Param as SystemParam>::State)>,
    /// `this_run` of the previous run; `None` before the first, which reports every change.
    last_run: Option<Tick>,
    _marker: PhantomData<fn() -> Marker>,
}

impl<Marker: 'static, F: SystemParamFunction<Marker>> System for FunctionSystem<Marker, F> {
    fn default_name(&self) -> &'static str {
        type_name::<F>()
    }

    fn run(&mut self, world: &mut World) {
        let world_id = world.id();
        let (state_world, state) = self.state.get_or_insert_with(|| {
            let mut access = SystemAccess::default();
            let state = F::Param::init_state(world, &mut access);
            (world_id, state)
        });
        assert_eq!(
            *state_world,
            world_id,
            "system `{}` run on a World it was not built for",
            type_name::<F>()
        );
        F::Param::update_state(state, world);
        let this_run = world.increment_change_tick();
        let mut last_run = self
            .last_run
            .unwrap_or(Tick::new(this_run.get().wrapping_sub(MAX_CHANGE_AGE)));
        last_run.clamp_age(this_run);
        let ticks = ChangeTicks { last_run, this_run };
        {
            let cell = WorldCell::new(world);
            // SAFETY: `state` was built for this world, and its accesses were checked for
            // conflicts when it was built.
            let param = unsafe { F::Param::get_param(state, cell, ticks) };
            self.func.run(param);
        }
        self.last_run = Some(this_run);
        F::Param::apply(state, world);
    }
}

/// An exclusive system: a function of `&mut World`.
pub struct ExclusiveSystem<F: FnMut(&mut World) + Send + 'static>(F);

impl<F: FnMut(&mut World) + Send + 'static> System for ExclusiveSystem<F> {
    fn default_name(&self) -> &'static str {
        type_name::<F>()
    }

    fn run(&mut self, world: &mut World) {
        (self.0)(world);
    }
}

/// Conversion of a function into a [`System`]; `Marker` tells the kinds of function apart.
pub trait IntoSystem<Marker>: Sized {
    /// The system type.
    type System: System;

    /// Converts the function.
    fn into_system(self) -> Self::System;
}

/// [`IntoSystem`] marker for functions with typed parameters.
pub struct ParamFunctionMarker;

/// [`IntoSystem`] marker for exclusive systems.
pub struct ExclusiveMarker;

impl<Marker: 'static, F: SystemParamFunction<Marker>> IntoSystem<(ParamFunctionMarker, Marker)>
    for F
{
    type System = FunctionSystem<Marker, F>;

    fn into_system(self) -> Self::System {
        FunctionSystem {
            func: self,
            state: None,
            last_run: None,
            _marker: PhantomData,
        }
    }
}

impl<F: FnMut(&mut World) + Send + 'static> IntoSystem<ExclusiveMarker> for F {
    type System = ExclusiveSystem<F>;

    fn into_system(self) -> Self::System {
        ExclusiveSystem(self)
    }
}
