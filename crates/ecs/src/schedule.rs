//! Stage-based system scheduling: named systems, ordering constraints and run conditions.
//!
//! A [`Schedule`] holds systems in five stages. [`Schedule::add`] takes any system: a function
//! with typed parameters ([`crate::system`]) or an exclusive `fn(&mut World)`;
//! [`Schedule::add_system`] takes an exclusive closure (its argument type can then stay
//! unannotated). Both return a [`SystemConfig`] to name the system, order it `before` or `after`
//! other systems of its stage by name, and give it a run condition. Within a stage, systems run
//! in an order that satisfies every constraint, otherwise in the order they were added; a cycle
//! panics when the stage first runs. Systems run one after another; the access each system
//! declares is what a later parallel runner needs.

use crate::system::{IntoSystem, System};
use crate::world::World;
use std::borrow::Cow;

/// Standard engine execution stages executed in deterministic sequential order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Initial stage for gathering OS inputs, events, and resetting scratch arenas.
    PreUpdate,
    /// Fixed-timestep stage executing deterministic physics updates and collision solvers.
    FixedUpdate,
    /// Primary variable-rate stage for gameplay logic, animations, and entity mutations.
    Update,
    /// Post-update stage for transform propagation, spatial partitioning, and culling.
    PostUpdate,
    /// Render extraction stage gathering render packets for GPU submission.
    Render,
}

impl Stage {
    const ALL: [Self; 5] = [
        Self::PreUpdate,
        Self::FixedUpdate,
        Self::Update,
        Self::PostUpdate,
        Self::Render,
    ];

    const fn index(self) -> usize {
        self as usize
    }
}

type RunCondition = Box<dyn FnMut(&World) -> bool + Send>;

struct SystemEntry {
    system: Box<dyn System>,
    name: Cow<'static, str>,
    before: Vec<Cow<'static, str>>,
    after: Vec<Cow<'static, str>>,
    run_if: Option<RunCondition>,
    enabled: bool,
}

#[derive(Default)]
struct StageSystems {
    entries: Vec<SystemEntry>,
    /// Indices into `entries` in run order; rebuilt when `dirty`.
    order: Vec<usize>,
    dirty: bool,
}

impl StageSystems {
    /// Orders the systems so that every `before`/`after` constraint holds, keeping insertion
    /// order among unconstrained systems. Constraints naming no system of the stage are ignored.
    fn resolve(&mut self, stage: Stage) {
        let n = self.entries.len();
        // `edges[a]` lists the systems that must run after `a`.
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indegree = vec![0usize; n];
        let named = |name: &str| -> Vec<usize> {
            self.entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.name == name)
                .map(|(i, _)| i)
                .collect()
        };
        for (i, entry) in self.entries.iter().enumerate() {
            for later in &entry.before {
                for j in named(later) {
                    edges[i].push(j);
                    indegree[j] += 1;
                }
            }
            for earlier in &entry.after {
                for j in named(earlier) {
                    edges[j].push(i);
                    indegree[i] += 1;
                }
            }
        }
        self.order.clear();
        let mut done = vec![false; n];
        for _ in 0..n {
            let Some(next) = (0..n).find(|&i| !done[i] && indegree[i] == 0) else {
                let cycle: Vec<&str> = (0..n)
                    .filter(|&i| !done[i])
                    .map(|i| self.entries[i].name.as_ref())
                    .collect();
                panic!("ordering cycle in stage {stage:?} between systems {cycle:?}");
            };
            done[next] = true;
            self.order.push(next);
            for &j in &edges[next] {
                indegree[j] -= 1;
            }
        }
        self.dirty = false;
    }
}

/// Manages and executes scheduled systems organized by execution stages.
#[derive(Default)]
pub struct Schedule {
    stages: [StageSystems; 5],
}

/// Configures a system just added to a [`Schedule`]; each method applies at once and returns
/// the configuration for chaining.
pub struct SystemConfig<'a> {
    stage: &'a mut StageSystems,
    index: usize,
}

impl SystemConfig<'_> {
    fn entry(&mut self) -> &mut SystemEntry {
        self.stage.dirty = true;
        &mut self.stage.entries[self.index]
    }

    /// Names the system (the default name is its function's type name).
    pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.entry().name = name.into();
        self
    }

    /// Runs the system before the systems of its stage named `other`.
    pub fn before(mut self, other: impl Into<Cow<'static, str>>) -> Self {
        self.entry().before.push(other.into());
        self
    }

    /// Runs the system after the systems of its stage named `other`.
    pub fn after(mut self, other: impl Into<Cow<'static, str>>) -> Self {
        self.entry().after.push(other.into());
        self
    }

    /// Runs the system only when `condition` returns `true` for the world.
    pub fn run_if(mut self, condition: impl FnMut(&World) -> bool + Send + 'static) -> Self {
        self.entry().run_if = Some(Box::new(condition));
        self
    }
}

impl Schedule {
    /// Creates a new, empty system schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a system to `stage`: a function with typed parameters, or an exclusive
    /// `fn(&mut World)`.
    pub fn add<M>(&mut self, stage: Stage, system: impl IntoSystem<M>) -> SystemConfig<'_> {
        self.push(stage, Box::new(system.into_system()))
    }

    /// Adds an exclusive system, a closure over `&mut World`, to `stage`.
    pub fn add_system<F>(&mut self, stage: Stage, system: F) -> SystemConfig<'_>
    where
        F: FnMut(&mut World) + Send + 'static,
    {
        self.add(stage, system)
    }

    fn push(&mut self, stage: Stage, system: Box<dyn System>) -> SystemConfig<'_> {
        let systems = &mut self.stages[stage.index()];
        systems.entries.push(SystemEntry {
            name: Cow::Borrowed(system.default_name()),
            system,
            before: Vec::new(),
            after: Vec::new(),
            run_if: None,
            enabled: true,
        });
        systems.dirty = true;
        let index = systems.entries.len() - 1;
        SystemConfig {
            stage: systems,
            index,
        }
    }

    /// Removes every system named `name`; returns `true` if there was one.
    pub fn remove(&mut self, name: &str) -> bool {
        let mut found = false;
        for systems in &mut self.stages {
            let before = systems.entries.len();
            systems.entries.retain(|entry| entry.name != name);
            if systems.entries.len() != before {
                found = true;
                systems.dirty = true;
            }
        }
        found
    }

    /// Enables or disables every system named `name`; returns `true` if there was one.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        let mut found = false;
        for systems in &mut self.stages {
            for entry in &mut systems.entries {
                if entry.name == name {
                    entry.enabled = enabled;
                    found = true;
                }
            }
        }
        found
    }

    /// Runs all systems within a specific stage against the world.
    ///
    /// # Panics
    /// If the stage's ordering constraints form a cycle.
    pub fn run_stage(&mut self, stage: Stage, world: &mut World) {
        let systems = &mut self.stages[stage.index()];
        if systems.dirty {
            systems.resolve(stage);
        }
        for &index in &systems.order {
            let entry = &mut systems.entries[index];
            if !entry.enabled {
                continue;
            }
            if let Some(condition) = &mut entry.run_if {
                if !condition(world) {
                    continue;
                }
            }
            entry.system.run(world);
        }
    }

    /// Executes the full main loop pipeline in standard order:
    /// `PreUpdate` -> `Update` -> `PostUpdate` -> `Render`.
    ///
    /// Note: `FixedUpdate` should be called via [`run_stage`](Self::run_stage) controlled by the `FixedClock`.
    pub fn run(&mut self, world: &mut World) {
        for stage in Stage::ALL {
            if stage != Stage::FixedUpdate {
                self.run_stage(stage, world);
            }
        }
    }
}
