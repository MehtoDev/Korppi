//! Deferred structural changes: a command buffer applied to the world after a system runs.
//!
//! [`Commands`] records spawns, despawns, component and pair changes and custom closures while a
//! system iterates its queries; the schedule applies them, in the order they were recorded, when
//! the system returns. Commands live in a byte buffer that keeps its storage, so recording them
//! allocates nothing once the buffer has grown to a tick's needs.

use crate::bundle::Bundle;
use crate::component::Component;
use crate::entity::{Entity, EntityManager};
use crate::relation::Relation;
use crate::world::World;
use std::mem::{size_of, MaybeUninit};

/// A deferred operation on the world.
pub trait Command: Send + 'static {
    /// Applies the operation.
    fn apply(self, world: &mut World);
}

impl<F: FnOnce(&mut World) + Send + 'static> Command for F {
    fn apply(self, world: &mut World) {
        self(world);
    }
}

/// What the queue stores in front of each command: how to apply (or drop) it and its size.
#[derive(Clone, Copy)]
struct CommandMeta {
    /// Reads the command at the pointer and applies it to the world, or drops it if there is
    /// none.
    run: unsafe fn(*mut MaybeUninit<u8>, Option<&mut World>),
    size: usize,
}

/// Reads a `C` from `ptr` and applies it to `world`, or drops it.
///
/// # Safety
/// `ptr` must point at a `C` written by [`CommandQueue::push`] that has not been read since.
unsafe fn run_command<C: Command>(ptr: *mut MaybeUninit<u8>, world: Option<&mut World>) {
    // SAFETY: the caller guarantees a `C` was written here (unaligned) and not read since.
    let command = unsafe { ptr.cast::<C>().read_unaligned() };
    match world {
        Some(world) => command.apply(world),
        None => drop(command),
    }
}

/// A sequence of type-erased commands in one byte buffer.
#[derive(Default)]
pub struct CommandQueue {
    bytes: Vec<MaybeUninit<u8>>,
}

// SAFETY: every command stored in `bytes` is `Send` (the `Command` bound), and the buffer is only
// ever read or written through `&mut CommandQueue`.
unsafe impl Send for CommandQueue {}
// SAFETY: a shared `&CommandQueue` exposes nothing but `is_empty`, which reads the length.
unsafe impl Sync for CommandQueue {}

impl CommandQueue {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if no command is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Records `command`.
    pub fn push<C: Command>(&mut self, command: C) {
        let meta = CommandMeta {
            run: run_command::<C>,
            size: size_of::<C>(),
        };
        let old_len = self.bytes.len();
        self.bytes
            .reserve(size_of::<CommandMeta>() + size_of::<C>());
        // SAFETY: the reservation above makes room for the meta and the command after `old_len`;
        // both are written unaligned into that spare capacity before the length covers them, and
        // `command` is moved into the buffer (not dropped here).
        unsafe {
            let end = self.bytes.as_mut_ptr().add(old_len);
            end.cast::<CommandMeta>().write_unaligned(meta);
            end.add(size_of::<CommandMeta>())
                .cast::<C>()
                .write_unaligned(command);
            self.bytes
                .set_len(old_len + size_of::<CommandMeta>() + size_of::<C>());
        }
    }

    /// Applies every command to `world` in the order they were recorded, then empties the queue
    /// (keeping its storage). Commands recorded by these commands run after them.
    pub fn apply(&mut self, world: &mut World) {
        world.flush_reserved_entities();
        // Commands may record further commands into the world's own queue; only this queue's
        // bytes are consumed here. The length drops to zero first, so a panicking command leaks
        // the rest instead of running or dropping any command twice.
        let len = self.bytes.len();
        // SAFETY: the bytes stay allocated; they are only read below.
        unsafe { self.bytes.set_len(0) };
        let base = self.bytes.as_mut_ptr();
        let mut offset = 0;
        while offset < len {
            // SAFETY: `push` wrote a meta at every command boundary, starting at 0.
            let meta = unsafe { base.add(offset).cast::<CommandMeta>().read_unaligned() };
            let command = offset + size_of::<CommandMeta>();
            offset = command + meta.size;
            // SAFETY: the command after this meta was written by `push` with this `run` function
            // and is read exactly once, here.
            unsafe { (meta.run)(base.add(command), Some(world)) };
        }
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        let len = self.bytes.len();
        // SAFETY: as in `apply`; each pending command is dropped exactly once.
        unsafe { self.bytes.set_len(0) };
        let base = self.bytes.as_mut_ptr();
        let mut offset = 0;
        while offset < len {
            // SAFETY: as in `apply`.
            let meta = unsafe { base.add(offset).cast::<CommandMeta>().read_unaligned() };
            let command = offset + size_of::<CommandMeta>();
            offset = command + meta.size;
            // SAFETY: as in `apply`; the command is dropped instead of applied.
            unsafe { (meta.run)(base.add(command), None) };
        }
    }
}

/// Records structural changes to apply to the world after the current system.
///
/// [`spawn`](Self::spawn) returns the new entity's handle at once; the entity exists (and is
/// alive) once the commands are applied.
pub struct Commands<'w, 's> {
    queue: &'s mut CommandQueue,
    entities: &'w EntityManager,
}

impl<'w, 's> Commands<'w, 's> {
    pub(crate) fn new(queue: &'s mut CommandQueue, entities: &'w EntityManager) -> Self {
        Self { queue, entities }
    }

    /// Records a spawn of an entity with `bundle` and returns the entity it will be.
    pub fn spawn<B: Bundle + Send>(&mut self, bundle: B) -> Entity {
        let entity = self.entities.reserve();
        self.queue.push(move |world: &mut World| {
            world.spawn_reserved(entity, bundle);
        });
        entity
    }

    /// Records a despawn of `entity` (with its cascade).
    pub fn despawn(&mut self, entity: Entity) {
        self.queue.push(move |world: &mut World| {
            world.despawn(entity);
        });
    }

    /// Records adding (or overwriting) `component` on `entity`; skipped if the entity is dead by
    /// then.
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        self.queue.push(move |world: &mut World| {
            if world.is_alive(entity) {
                world.add_component(entity, component);
            }
        });
    }

    /// Records removing `T` from `entity`.
    pub fn remove<T: Component>(&mut self, entity: Entity) {
        self.queue.push(move |world: &mut World| {
            world.remove_component::<T>(entity);
        });
    }

    /// Records adding the pair `(R, target)` to `entity`; skipped if either entity is dead by
    /// then or the pair would create a cycle.
    pub fn add_pair<R: Relation>(&mut self, entity: Entity, target: Entity) {
        self.queue.push(move |world: &mut World| {
            let _ = world.try_add_pair::<R>(entity, target);
        });
    }

    /// Records removing the pair `(R, target)` from `entity`.
    pub fn remove_pair<R: Relation>(&mut self, entity: Entity, target: Entity) {
        self.queue.push(move |world: &mut World| {
            world.remove_pair::<R>(entity, target);
        });
    }

    /// Records a custom command.
    pub fn add<C: Command>(&mut self, command: C) {
        self.queue.push(command);
    }
}
