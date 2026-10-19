//! Running tasks on several threads: the task system Box2D v3.2 asks its host for
//! (`b2EnqueueTaskCallback`, `b2FinishTaskCallback`), as a [`TaskExecutor`] trait with a
//! [`SerialExecutor`] and a [`WorkerPool`] of persistent threads.
//!
//! [`TaskExecutor::run`] calls a task for each index of a range, spread over the workers, the
//! calling thread among them, and returns once every call has returned. It does not allocate,
//! so a simulation can use it every tick. Tasks of one run may wait for each other (a solver's
//! workers spin on a shared stage counter), as long as each can finish once the others have
//! started: a [`WorkerPool`] gives every task of a run its own thread when the run has no more
//! tasks than workers.
//!
//! [`parallel_for`] splits a range of items into blocks over an executor, and [`spin_until`]
//! waits for other workers the way Box2D's solver workers do.

use std::any::Any;
use std::cell::UnsafeCell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// Runs tasks for a caller on its workers (the host side of Box2D's task callbacks).
pub trait TaskExecutor: Send + Sync {
    /// Number of workers, the calling thread included: at least 1.
    fn worker_count(&self) -> usize;

    /// Calls `task(index, worker)` for every `index` in `0..task_count` and returns when every
    /// call has returned. The calling thread runs tasks too. `worker` is below
    /// [`Self::worker_count`], and two calls running at the same time never share one, so a
    /// task may keep per-worker scratch. Tasks run in any order. A task must not call `run` on
    /// the same executor.
    fn run(&self, task_count: usize, task: &(dyn Fn(usize, usize) + Sync));
}

/// Runs every task on the calling thread, in order.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerialExecutor;

impl TaskExecutor for SerialExecutor {
    fn worker_count(&self) -> usize {
        1
    }

    fn run(&self, task_count: usize, task: &(dyn Fn(usize, usize) + Sync)) {
        for index in 0..task_count {
            task(index, 0);
        }
    }
}

/// Calls `task(start, end, worker)` over `0..item_count` in blocks of at least `min_range`
/// items, at most eight per worker, spread over `executor` (Box2D `b2ParallelFor`). Workers
/// claim blocks in order through an atomic counter, and the calling thread finishes any block
/// left unclaimed, so every item is visited exactly once whatever the executor does. A task
/// that writes only its own items gives the same result on any executor.
pub fn parallel_for(
    executor: &dyn TaskExecutor,
    item_count: usize,
    min_range: usize,
    task: &(dyn Fn(usize, usize, usize) + Sync),
) {
    if item_count == 0 {
        return;
    }
    let workers = executor.worker_count().max(1);
    let max_blocks = 8usize.saturating_mul(workers);
    let min_range = min_range.max(1);
    let block_size = if item_count <= min_range.saturating_mul(max_blocks) {
        min_range
    } else {
        item_count.div_ceil(max_blocks)
    };
    let block_count = item_count.div_ceil(block_size);
    if block_count == 1 || workers == 1 {
        task(0, item_count, 0);
        return;
    }
    let next = AtomicUsize::new(0);
    let claim = |worker: usize| loop {
        let block = next.fetch_add(1, Ordering::Relaxed);
        if block >= block_count {
            return;
        }
        let start = block * block_size;
        task(
            start,
            start.saturating_add(block_size).min(item_count),
            worker,
        );
    };
    executor.run(workers.min(block_count), &|_, worker| claim(worker));
    claim(0);
}

/// Waits for `done` to hold, spinning between checks and yielding the thread only if the wait
/// drags on (Box2D's worker spin: `b2Pause`, then `b2Yield`).
pub fn spin_until(mut done: impl FnMut() -> bool) {
    let mut spins = 0u32;
    while !done() {
        if spins < 2048 {
            spins += 1;
            std::hint::spin_loop();
        } else {
            spins = 0;
            std::thread::yield_now();
        }
    }
}

/// The task of the current run, its lifetime erased: [`WorkerPool::run`] keeps it alive
/// until no worker can read it.
type ErasedTask = *const (dyn Fn(usize, usize) + Sync + 'static);

/// A run the pool's threads work on.
#[derive(Clone, Copy)]
struct Job {
    task: ErasedTask,
    count: usize,
}

// Packing of `Shared::status`:
// - bits `0..16`: number of worker threads currently inside `work()` (`BUSY_MASK`)
// - bit `16`: `ACTIVE_BIT` (1 while workers may join the current run)
// - bit `17`: `SHUTDOWN_BIT` (1 when the pool is dropping)
// - bits `18..64`: `run_id` (increments by 1 on each `WorkerPool::run`)
const BUSY_MASK: u64 = (1 << 16) - 1;
const ACTIVE_BIT: u64 = 1 << 16;
const SHUTDOWN_BIT: u64 = 1 << 17;
const RUN_ID_SHIFT: u32 = 18;
const RUN_ID_INC: u64 = 1 << RUN_ID_SHIFT;

/// Spin iterations before a worker thread parks on the condition variable (~100-150 us,
/// keeping workers awake across the narrowphase, solver, and body-finalization stages of a
/// single physics step, as Box2D's `B2_SCHEDULER_SPIN_MILLISECONDS`).
const WORKER_SPIN_ITERS: u32 = 32_768;

/// State shared between a pool and its threads.
struct Shared {
    /// Packed `(run_id, SHUTDOWN_BIT, ACTIVE_BIT, busy)`.
    status: AtomicU64,
    /// The current run's job descriptor. Written by the caller while `ACTIVE_BIT == 0` and
    /// `busy == 0`; read by workers only after incrementing `busy` while `ACTIVE_BIT == 1`.
    job: UnsafeCell<Job>,
    /// Number of worker threads currently parked (or entering wait) on `wake`.
    sleeping: AtomicUsize,
    /// Guards `wake` so wakeups cannot be lost between checking `status` and parking.
    sleep_lock: Mutex<()>,
    /// Wakes parked threads for a new run or for shutdown.
    wake: Condvar,
    /// The next task index to claim.
    next: AtomicUsize,
    /// Tasks of the current run that have not returned.
    remaining: AtomicUsize,
    /// Set when a task in the current run panics so the non-panicking fast path skips locking
    /// `panic`.
    has_panic: AtomicBool,
    /// The first panic of the current run's tasks, raised again on the caller.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

// SAFETY: `Shared::job` is only written by the single `run` caller while `ACTIVE_BIT == 0` and
// `busy == 0`, and only read by workers while `busy > 0` after acquiring `status`. `Job::task`
// points to a `Sync` closure kept alive by `WorkerPool::run` until `busy == 0`.
unsafe impl Sync for Shared {}
// SAFETY: see `Sync` above.
unsafe impl Send for Shared {}

impl Shared {
    /// Claims and runs tasks of `job` as worker `worker` until none is left.
    fn work(&self, job: Job, worker: usize) {
        loop {
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            if index >= job.count {
                return;
            }
            // SAFETY: the job was read while `ACTIVE_BIT` was set and `busy > 0`, and `run`
            // does not return or unwind before `busy` reaches 0, so the closure is alive.
            let task = unsafe { &*job.task };
            // A panicking task still counts as returned, so that the run ends; its panic is
            // raised again on the caller.
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| task(index, worker))) {
                let mut panic = self.panic.lock().unwrap_or_else(PoisonError::into_inner);
                if panic.is_none() {
                    *panic = Some(payload);
                    self.has_panic.store(true, Ordering::Release);
                }
            }
            self.remaining.fetch_sub(1, Ordering::Release);
        }
    }
}

/// A fixed set of worker threads that run the tasks of one [`TaskExecutor::run`] at a time,
/// with the calling thread as worker 0. Runs from several threads take turns. A task that
/// panics does not stop the others; once all have returned, `run` raises the first panic on
/// the calling thread.
pub struct WorkerPool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    /// Serializes runs.
    turn: Mutex<()>,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("worker_count", &self.worker_count())
            .finish()
    }
}

impl WorkerPool {
    /// A pool of `worker_count` workers: the calling thread and `worker_count - 1` threads
    /// (at least one worker).
    ///
    /// # Panics
    /// Panics if `worker_count > 65_535`.
    #[must_use]
    pub fn new(worker_count: usize) -> Self {
        assert!(
            worker_count <= BUSY_MASK as usize,
            "WorkerPool worker_count ({worker_count}) exceeds maximum supported workers ({BUSY_MASK})"
        );
        let nop: &(dyn Fn(usize, usize) + Sync + 'static) = &|_, _| {};
        let shared = Arc::new(Shared {
            status: AtomicU64::new(0),
            job: UnsafeCell::new(Job {
                task: nop as *const _,
                count: 0,
            }),
            sleeping: AtomicUsize::new(0),
            sleep_lock: Mutex::new(()),
            wake: Condvar::new(),
            next: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            has_panic: AtomicBool::new(false),
            panic: Mutex::new(None),
        });
        let threads = (1..worker_count.max(1))
            .map(|worker| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("korppi-worker-{worker}"))
                    .spawn(move || worker_loop(&shared, worker))
                    .expect("spawning a worker thread")
            })
            .collect();
        Self {
            shared,
            threads,
            turn: Mutex::new(()),
        }
    }
}

/// A pool thread: spins briefly between runs, parks when idle, and works on runs until shutdown.
fn worker_loop(shared: &Shared, worker: usize) {
    let mut seen_run = 0u64;
    let mut spins = 0u32;
    loop {
        let mut s = shared.status.load(Ordering::Acquire);
        if s & SHUTDOWN_BIT != 0 {
            return;
        }
        let run_id = s >> RUN_ID_SHIFT;
        if (s & ACTIVE_BIT != 0) && run_id != seen_run {
            loop {
                match shared.status.compare_exchange_weak(
                    s,
                    s + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        seen_run = run_id;
                        spins = 0;
                        // SAFETY: `busy` is now > 0 for `run_id`, and the caller does not
                        // mutate `shared.job` while `ACTIVE_BIT != 0` or `busy != 0`.
                        let job = unsafe { *shared.job.get() };
                        shared.work(job, worker);
                        shared.status.fetch_sub(1, Ordering::Release);
                        break;
                    }
                    Err(actual) => {
                        s = actual;
                        if s & SHUTDOWN_BIT != 0 {
                            return;
                        }
                        if (s & ACTIVE_BIT == 0) || (s >> RUN_ID_SHIFT) != run_id {
                            break;
                        }
                    }
                }
            }
            continue;
        }

        if spins < WORKER_SPIN_ITERS {
            spins += 1;
            std::hint::spin_loop();
            continue;
        }

        spins = 0;
        shared.sleeping.fetch_add(1, Ordering::SeqCst);
        let mut guard = shared
            .sleep_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        loop {
            let cur = shared.status.load(Ordering::SeqCst);
            if (cur & SHUTDOWN_BIT != 0)
                || ((cur & ACTIVE_BIT != 0) && (cur >> RUN_ID_SHIFT) != seen_run)
            {
                break;
            }
            guard = shared
                .wake
                .wait(guard)
                .unwrap_or_else(PoisonError::into_inner);
        }
        drop(guard);
        shared.sleeping.fetch_sub(1, Ordering::SeqCst);
    }
}

impl TaskExecutor for WorkerPool {
    fn worker_count(&self) -> usize {
        self.threads.len() + 1
    }

    fn run(&self, task_count: usize, task: &(dyn Fn(usize, usize) + Sync)) {
        if task_count == 0 {
            return;
        }
        if self.threads.is_empty() || task_count == 1 {
            return SerialExecutor.run(task_count, task);
        }
        let _turn = self.turn.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: only the lifetime is erased. The pointer is published below and withdrawn
        // before this function returns, after every thread that read it has left the run, so
        // no thread calls the closure after it is gone.
        let erased: ErasedTask =
            unsafe { std::mem::transmute::<&(dyn Fn(usize, usize) + Sync), ErasedTask>(task) };
        let job = Job {
            task: erased,
            count: task_count,
        };
        // SAFETY: `_turn` serializes callers, and the previous `run` waited until `ACTIVE_BIT`
        // was cleared and `busy == 0`, so no worker thread is reading `shared.job`.
        unsafe {
            *self.shared.job.get() = job;
        }
        self.shared.next.store(0, Ordering::Relaxed);
        self.shared.remaining.store(task_count, Ordering::Relaxed);
        self.shared
            .status
            .fetch_add(RUN_ID_INC | ACTIVE_BIT, Ordering::SeqCst);
        if self.shared.sleeping.load(Ordering::SeqCst) != 0 {
            let _guard = self
                .shared
                .sleep_lock
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            self.shared.wake.notify_all();
        }
        self.shared.work(job, 0);
        // Every task is claimed; wait for the last ones, then clear `ACTIVE_BIT` and wait for
        // every worker thread that joined this run to leave `work()`.
        spin_until(|| self.shared.remaining.load(Ordering::Acquire) == 0);
        self.shared.status.fetch_and(!ACTIVE_BIT, Ordering::Acquire);
        while self.shared.status.load(Ordering::Acquire) & BUSY_MASK != 0 {
            std::hint::spin_loop();
        }
        if self.shared.has_panic.swap(false, Ordering::AcqRel) {
            let panic = self
                .shared
                .panic
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(payload) = panic {
                resume_unwind(payload);
            }
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shared.status.fetch_or(SHUTDOWN_BIT, Ordering::SeqCst);
        let _guard = self
            .shared
            .sleep_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.shared.wake.notify_all();
        drop(_guard);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
