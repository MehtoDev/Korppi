//! `TaskExecutor` runs every task of a run exactly once, on exclusive workers, and returns only
//! when all are done; a `WorkerPool` runs them on its threads, lets the tasks of one run wait
//! for each other, and does not allocate per run (Box2D's task system contract, which its
//! parallel solver relies on).

use korppi_core::{parallel_for, SerialExecutor, TaskExecutor, WorkerPool};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

/// Counts allocations on threads that opted in.
struct CountingAllocator;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

fn count() {
    let _ = TRACKING.try_with(|tracking| {
        if tracking.get() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
    });
}

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract. Counting only touches a const-initialized `Cell` thread-local and an
// atomic, which never allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged; the caller upholds `dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn executors() -> Vec<Box<dyn TaskExecutor>> {
    vec![
        Box::new(SerialExecutor),
        Box::new(WorkerPool::new(1)),
        Box::new(WorkerPool::new(2)),
        Box::new(WorkerPool::new(4)),
        Box::new(WorkerPool::new(7)),
    ]
}

#[test]
fn every_task_runs_once_on_an_exclusive_worker() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    for executor in executors() {
        let workers = executor.worker_count();
        for task_count in [0, 1, 2, 3, 10, 1000] {
            let runs: Vec<AtomicUsize> = (0..task_count).map(|_| AtomicUsize::new(0)).collect();
            let inside: Vec<AtomicBool> = (0..workers).map(|_| AtomicBool::new(false)).collect();
            executor.run(task_count, &|index, worker| {
                assert!(worker < workers);
                assert!(
                    !inside[worker].swap(true, Ordering::AcqRel),
                    "worker {worker} shared"
                );
                runs[index].fetch_add(1, Ordering::Relaxed);
                inside[worker].store(false, Ordering::Release);
            });
            assert!(runs.iter().all(|r| r.load(Ordering::Relaxed) == 1));
        }
    }
}

#[test]
fn tasks_of_a_run_may_wait_for_each_other() {
    // As Box2D's solver: every task waits until all have started.
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    for workers in [2, 3, 4, 8] {
        let pool = WorkerPool::new(workers);
        for _ in 0..50 {
            let started = AtomicUsize::new(0);
            pool.run(workers, &|_, _| {
                started.fetch_add(1, Ordering::AcqRel);
                while started.load(Ordering::Acquire) < workers {
                    std::hint::spin_loop();
                }
            });
        }
    }
}

/// An executor that claims four workers but runs no task.
struct Idle;

impl TaskExecutor for Idle {
    fn worker_count(&self) -> usize {
        4
    }

    fn run(&self, _: usize, _: &(dyn Fn(usize, usize) + Sync)) {}
}

/// An executor that runs every task twice, in reverse order.
struct Twice;

impl TaskExecutor for Twice {
    fn worker_count(&self) -> usize {
        3
    }

    fn run(&self, task_count: usize, task: &(dyn Fn(usize, usize) + Sync)) {
        for index in (0..task_count).rev() {
            task(index, 0);
            task(index, 0);
        }
    }
}

#[test]
fn parallel_for_covers_the_range_in_blocks_on_any_executor() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let mut executors = executors();
    executors.push(Box::new(Idle));
    executors.push(Box::new(Twice));
    for executor in executors {
        for (items, min_range) in [(0, 1), (1, 1), (5, 2), (100, 1), (1000, 64), (7, 100)] {
            let seen: Vec<AtomicUsize> = (0..items).map(|_| AtomicUsize::new(0)).collect();
            parallel_for(executor.as_ref(), items, min_range, &|start, end, _| {
                assert!(start < end && end <= items);
                assert!(end - start >= min_range.min(items - start));
                for item in &seen[start..end] {
                    item.fetch_add(1, Ordering::Relaxed);
                }
            });
            assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1));
        }
    }
}

#[test]
fn a_pool_run_does_not_allocate() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let workers = 4;
    let pool = WorkerPool::new(workers);
    // Every thread of the pool, and this one, counts its allocations from here on: one task
    // per worker, each waiting for all to start, puts a task on every thread.
    let started = AtomicUsize::new(0);
    pool.run(workers, &|_, _| {
        TRACKING.with(|t| t.set(true));
        started.fetch_add(1, Ordering::AcqRel);
        while started.load(Ordering::Acquire) < workers {
            std::hint::spin_loop();
        }
    });
    let sum = AtomicUsize::new(0);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    for _ in 0..100 {
        pool.run(16, &|index, _| {
            sum.fetch_add(index, Ordering::Relaxed);
        });
        parallel_for(&pool, 1000, 10, &|start, end, _| {
            sum.fetch_add(end - start, Ordering::Relaxed);
        });
    }
    TRACKING.with(|t| t.set(false));
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert_eq!(sum.load(Ordering::Relaxed), 100 * (120 + 1000));
}

#[test]
fn a_panicking_task_reaches_the_caller_and_the_pool_carries_on() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let pool = WorkerPool::new(4);
    let runs = AtomicUsize::new(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.run(100, &|index, _| {
            runs.fetch_add(1, Ordering::Relaxed);
            assert_ne!(index, 37, "task 37 fails");
        });
    }));
    let message = result.expect_err("the panic reaches the caller");
    assert!(message
        .downcast_ref::<String>()
        .is_some_and(|m| m.contains("task 37 fails")));
    // Every task ran, and the pool still works.
    assert_eq!(runs.load(Ordering::Relaxed), 100);
    let sum = AtomicUsize::new(0);
    pool.run(10, &|index, _| {
        sum.fetch_add(index, Ordering::Relaxed);
    });
    assert_eq!(sum.load(Ordering::Relaxed), 45);
}

#[test]
fn back_to_back_runs_complete_without_sleep_stall() {
    // Between `collide`, `solve`, and `finalize_bodies` in a single physics step, the calling
    // thread does a few microseconds of serial bookkeeping. Workers must spin briefly before
    // parking on the condition variable so back-to-back runs within a step do not pay OS
    // futex wake/wait latency on every run.
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let pool = WorkerPool::new(4);
    let counter = AtomicUsize::new(0);
    pool.run(4, &|_, _| {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    let start = std::time::Instant::now();
    for _ in 0..500 {
        // Simulate ~5 us of serial step bookkeeping between parallel stages.
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_nanos() < 5_000 {
            std::hint::spin_loop();
        }
        let entered = AtomicUsize::new(0);
        pool.run(4, &|index, _| {
            entered.fetch_add(1, Ordering::AcqRel);
            while entered.load(Ordering::Acquire) < 4 {
                std::hint::spin_loop();
            }
            counter.fetch_add(index + 1, Ordering::Relaxed);
        });
    }
    let elapsed = start.elapsed();
    assert_eq!(counter.load(Ordering::Relaxed), 4 + 500 * 10);
    assert!(
        elapsed.as_millis() < 10,
        "500 4-worker runs separated by 5 us of serial work must not stall on condvar sleep/wake, took {:?}",
        elapsed
    );
}
