//! Observers: callbacks that run when a component is added, set or removed (ADR-048).
//!
//! [`World::observe`](crate::World::observe) registers a callback for an [`OnAdd<T>`],
//! [`OnSet<T>`] or [`OnRemove<T>`] event. The world records events while a structural change runs
//! and calls the observers once the change is complete, with `&mut World` and the entity:
//! - `OnAdd<T>`: `T` was added to the entity (spawned with it, or added to it);
//! - `OnSet<T>`: a `T` value was written by spawning, adding or overwriting;
//! - `OnRemove<T>`: `T` was removed from the entity, or the entity was despawned with it (the
//!   value is gone by the time the observer runs).
//!
//! Events are handled in the order they happened, observers in the order they were registered.
//! Changes an observer makes record further events, handled after the ones already waiting.
//! Recording an event allocates nothing once the event buffer has grown.

use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::world::World;
use std::marker::PhantomData;

/// The kind of change an observer reacts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// A component was added.
    Add,
    /// A component value was written.
    Set,
    /// A component was removed.
    Remove,
}

impl EventKind {
    /// This kind's bit in an observed-events mask.
    pub(crate) const fn bit(self) -> u8 {
        match self {
            Self::Add => 1,
            Self::Set => 2,
            Self::Remove => 4,
        }
    }
}

/// An event type for [`World::observe`](crate::World::observe).
pub trait ObserverEvent: 'static {
    /// The component the event concerns.
    type Component: Component;
    /// The kind of change.
    const KIND: EventKind;
}

/// `T` was added to an entity.
pub struct OnAdd<T: Component>(PhantomData<T>);

/// A `T` value was written to an entity.
pub struct OnSet<T: Component>(PhantomData<T>);

/// `T` was removed from an entity, or the entity was despawned.
pub struct OnRemove<T: Component>(PhantomData<T>);

impl<T: Component> ObserverEvent for OnAdd<T> {
    type Component = T;
    const KIND: EventKind = EventKind::Add;
}

impl<T: Component> ObserverEvent for OnSet<T> {
    type Component = T;
    const KIND: EventKind = EventKind::Set;
}

impl<T: Component> ObserverEvent for OnRemove<T> {
    type Component = T;
    const KIND: EventKind = EventKind::Remove;
}

/// Identifies an observer for [`World::remove_observer`](crate::World::remove_observer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObserverId(u64);

type Callback = Box<dyn FnMut(&mut World, Entity) + Send>;

struct Observer {
    id: ObserverId,
    kind: EventKind,
    component: ComponentId,
    callback: Callback,
}

/// An event waiting for its observers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingEvent {
    pub(crate) kind: EventKind,
    pub(crate) component: ComponentId,
    pub(crate) entity: Entity,
}

/// The world's observers and the events waiting for them.
#[derive(Default)]
pub(crate) struct Observers {
    entries: Vec<Observer>,
    /// Per component ID (by index): the event kinds some observer watches.
    mask: Vec<u8>,
    pending: Vec<PendingEvent>,
    /// Set while observers run (their list is taken out), so that nested changes only record
    /// their events.
    flushing: bool,
    /// Every observer removed so far.
    removed: Vec<ObserverId>,
    /// Set when observers were added or removed while observers ran.
    changed_while_flushing: bool,
    next_id: u64,
}

impl Observers {
    /// The event kinds observed for `component`.
    #[inline]
    pub(crate) fn mask(&self, component: ComponentId) -> u8 {
        self.mask.get(component.index()).copied().unwrap_or(0)
    }

    /// Records an event if anything observes it.
    #[inline]
    pub(crate) fn record(&mut self, kind: EventKind, component: ComponentId, entity: Entity) {
        if self.mask(component) & kind.bit() != 0 {
            self.pending.push(PendingEvent {
                kind,
                component,
                entity,
            });
        }
    }

    fn add(&mut self, kind: EventKind, component: ComponentId, callback: Callback) -> ObserverId {
        let id = ObserverId(self.next_id);
        self.next_id += 1;
        self.entries.push(Observer {
            id,
            kind,
            component,
            callback,
        });
        // Adding only sets bits, so the mask stays right while observers run.
        let index = component.index();
        if self.mask.len() <= index {
            self.mask.resize(index + 1, 0);
        }
        self.mask[index] |= kind.bit();
        self.changed_while_flushing |= self.flushing;
        id
    }

    fn remove(&mut self, id: ObserverId) -> bool {
        if id.0 >= self.next_id || self.removed.contains(&id) {
            return false;
        }
        self.removed.push(id);
        if self.flushing {
            // The running list is out of reach; it is filtered when the observers finish.
            self.changed_while_flushing = true;
        } else {
            self.entries.retain(|o| o.id != id);
            self.rebuild_mask();
        }
        true
    }

    fn rebuild_mask(&mut self) {
        self.mask.iter_mut().for_each(|bits| *bits = 0);
        for observer in &self.entries {
            let index = observer.component.index();
            if self.mask.len() <= index {
                self.mask.resize(index + 1, 0);
            }
            self.mask[index] |= observer.kind.bit();
        }
    }
}

impl World {
    /// Registers `observer` for the event `E`, such as `OnAdd<RigidBody>`. It runs after each
    /// change that causes the event, with the world and the entity.
    pub fn observe<E: ObserverEvent>(
        &mut self,
        observer: impl FnMut(&mut World, Entity) + Send + 'static,
    ) -> ObserverId {
        let component = self.register_component::<E::Component>();
        let id = self
            .observers_mut()
            .add(E::KIND, component, Box::new(observer));
        self.refresh_observed_masks();
        id
    }

    /// Unregisters an observer; returns `false` if it was not registered or is already removed.
    /// An observer removed while observers run stops once they finish.
    pub fn remove_observer(&mut self, id: ObserverId) -> bool {
        let removed = self.observers_mut().remove(id);
        if removed {
            self.refresh_observed_masks();
        }
        removed
    }

    /// Runs the observers of the events recorded so far, and of the events those observers
    /// cause, until none is left. Does nothing while observers are already running.
    pub(crate) fn flush_observers(&mut self) {
        let observers = self.observers_mut();
        if observers.pending.is_empty() || observers.flushing {
            return;
        }
        observers.flushing = true;
        let mut running = std::mem::take(&mut observers.entries);
        let mut next = 0;
        while let Some(&event) = self.observers_mut().pending.get(next) {
            next += 1;
            for observer in &mut running {
                if observer.kind == event.kind && observer.component == event.component {
                    (observer.callback)(self, event.entity);
                }
            }
        }
        let observers = self.observers_mut();
        observers.pending.clear();
        observers.flushing = false;
        let registered_meanwhile = std::mem::replace(&mut observers.entries, running);
        observers.entries.extend(registered_meanwhile);
        if std::mem::take(&mut observers.changed_while_flushing) {
            let removed = std::mem::take(&mut observers.removed);
            observers.entries.retain(|o| !removed.contains(&o.id));
            observers.removed = removed;
            observers.rebuild_mask();
            self.refresh_observed_masks();
        }
    }
}
