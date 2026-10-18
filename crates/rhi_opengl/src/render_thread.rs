//! Dedicated OpenGL render thread (findings RGL-08, RGL-11).
//!
//! An OpenGL context is current on at most one thread, and every GL call goes to the context
//! current on the calling thread. `GraphicsDevice` is `Send + Sync`, so instead of making the
//! context current on whichever thread calls in, `OpenGlDevice` owns one thread that creates the
//! context, makes it current once, executes every GL call and destroys it again.
//!
//! Device methods reach that thread in two ways:
//! - [`RenderThread::call`] runs a closure on the render thread and blocks until it returns. The
//!   closure may borrow from the caller (for example `initial_data`), so nothing is copied or
//!   boxed per call.
//! - [`RenderThread::post`] queues owned work (a submitted command buffer) and returns at once.
//!
//! Messages are handled strictly in the order they were sent, so a call always observes every
//! submission posted before it. Only the render thread touches the state it owns, which removes
//! the context mutex and with it the lock-order inversions of RGL-11.

use korppi_rhi::RhiError;
use std::panic::{self, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::thread::{self, JoinHandle, Thread, ThreadId};
use std::time::Duration;

/// Messages that may wait in the queue before a sender blocks (back-pressure on `post`).
const QUEUE_CAPACITY: usize = 16;

/// State owned by a render thread. It is created on that thread, used only there and dropped
/// there, so it need not be `Send`.
pub(crate) trait ThreadState: 'static {
    /// Work handed to the thread without a waiting caller.
    type Posted: Send + 'static;

    /// Handles one posted message.
    fn handle_posted(&mut self, posted: Self::Posted);

    /// Called before the thread waits for the next message. Returns how long to wait before
    /// calling it again if no message arrives (`None`: wait until one does).
    fn poll(&mut self) -> Option<Duration> {
        None
    }

    /// Records a panic raised by [`ThreadState::handle_posted`]; the thread keeps running.
    fn posted_panicked(&mut self, message: String);
}

/// Wakes a caller blocked in [`RenderThread::call`].
struct Completion {
    done: AtomicBool,
    caller: Thread,
}

/// A closure borrowed from a caller blocked in [`RenderThread::call`].
struct CallJob<S> {
    /// The caller's closure with its lifetime erased. Valid until `completion` is signalled.
    job: NonNull<dyn FnMut(&mut S) + Send>,
    /// Lives on the caller's stack. Valid until it is signalled.
    completion: NonNull<Completion>,
}

// SAFETY: a `CallJob` only carries pointers into the stack frame of a caller that stays blocked
// until the job signals completion (see `RenderThread::call`). The closure behind `job` is
// `Send`, and `Completion` holds only an atomic and a `Thread` handle, both `Send + Sync`.
unsafe impl<S> Send for CallJob<S> {}

impl<S> CallJob<S> {
    /// Runs the caller's closure on the render thread; completion is signalled when `self` drops.
    fn run(self, state: &mut S) {
        // SAFETY: the caller is blocked until completion is signalled, which happens only when
        // `self` drops (after this call returns), so the closure is alive and not aliased.
        unsafe { (*self.job.as_ptr())(state) };
    }
}

impl<S> Drop for CallJob<S> {
    /// Signals completion whether or not the job ran: a job dropped unrun (for example when the
    /// render thread exits with messages still queued) must not leave its caller blocked.
    fn drop(&mut self) {
        // SAFETY: the completion lives on the stack of a caller blocked until `done` is set.
        let completion = unsafe { self.completion.as_ref() };
        // Clone the handle first: once `done` is set the caller may return and free
        // `completion`, so it must not be touched afterwards.
        let caller = completion.caller.clone();
        completion.done.store(true, Ordering::Release);
        caller.unpark();
    }
}

enum Message<S: ThreadState> {
    Call(CallJob<S>),
    Post(S::Posted),
}

/// A thread that owns state of type `S` and runs work on it (see the module documentation).
pub(crate) struct RenderThread<S: ThreadState> {
    sender: Option<SyncSender<Message<S>>>,
    handle: Option<JoinHandle<()>>,
    thread_id: ThreadId,
}

impl<S: ThreadState> RenderThread<S> {
    /// Spawns a thread named `name` and builds its state there with `init`.
    ///
    /// Returns `init`'s error if it fails; the thread has exited by then.
    pub(crate) fn spawn(
        name: &str,
        init: impl FnOnce() -> Result<S, RhiError> + Send + 'static,
    ) -> Result<Self, RhiError> {
        let (sender, receiver) = mpsc::sync_channel::<Message<S>>(QUEUE_CAPACITY);
        let (ready_sender, ready) = mpsc::sync_channel::<Result<(), RhiError>>(1);
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let state = match panic::catch_unwind(AssertUnwindSafe(init)) {
                    Ok(Ok(state)) => state,
                    Ok(Err(error)) => {
                        let _ = ready_sender.send(Err(error));
                        return;
                    }
                    Err(payload) => {
                        let _ = ready_sender.send(Err(RhiError::InitializationFailed(format!(
                            "render thread initialization panicked: {}",
                            panic_message(&*payload)
                        ))));
                        return;
                    }
                };
                let _ = ready_sender.send(Ok(()));
                run(state, &receiver);
            })
            .map_err(|e| {
                RhiError::InitializationFailed(format!("failed to spawn the render thread: {e}"))
            })?;
        let thread_id = handle.thread().id();
        match ready.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Some(sender),
                handle: Some(handle),
                thread_id,
            }),
            Ok(Err(error)) => {
                let _ = handle.join();
                Err(error)
            }
            Err(_) => {
                let _ = handle.join();
                Err(RhiError::InitializationFailed(
                    "render thread exited during initialization".into(),
                ))
            }
        }
    }

    /// Runs `f` on the render thread and returns its result, blocking until it has run.
    ///
    /// A panic in `f` is resumed on the calling thread; the render thread keeps running.
    ///
    /// # Errors
    /// `RhiError::BackendError` if the render thread has exited, or if this is called from the
    /// render thread itself (which would deadlock).
    pub(crate) fn call<R: Send>(&self, f: impl FnOnce(&mut S) -> R + Send) -> Result<R, RhiError> {
        if thread::current().id() == self.thread_id {
            return Err(RhiError::BackendError(
                "RenderThread::call from the render thread itself would deadlock".into(),
            ));
        }
        let sender = self.sender()?;
        let mut f = Some(f);
        let mut result: Option<thread::Result<R>> = None;
        let mut job = |state: &mut S| {
            if let Some(f) = f.take() {
                result = Some(panic::catch_unwind(AssertUnwindSafe(|| f(state))));
            }
        };
        let completion = Completion {
            done: AtomicBool::new(false),
            caller: thread::current(),
        };
        let job: &mut (dyn FnMut(&mut S) + Send + '_) = &mut job;
        // SAFETY: only the lifetime bound of the trait object changes; the layout is identical.
        // The erased borrow is used by the render thread only until it signals `completion`, and
        // this function does not return or unwind before then (the wait loop below cannot panic).
        let job: NonNull<dyn FnMut(&mut S) + Send + 'static> =
            unsafe { std::mem::transmute(NonNull::from(job)) };
        let message = Message::Call(CallJob {
            job,
            completion: NonNull::from(&completion),
        });
        if let Err(mpsc::SendError(message)) = sender.send(message) {
            // Never handed over. Dropping the job only signals `completion`, which is still
            // alive here.
            drop(message);
            return Err(thread_exited());
        }
        while !completion.done.load(Ordering::Acquire) {
            thread::park();
        }
        match result {
            Some(Ok(value)) => Ok(value),
            Some(Err(payload)) => panic::resume_unwind(payload),
            None => Err(thread_exited()),
        }
    }

    /// Queues `posted` for the render thread and returns without waiting for it to be handled.
    /// Blocks only while the queue is full.
    pub(crate) fn post(&self, posted: S::Posted) -> Result<(), RhiError> {
        self.sender()?
            .send(Message::Post(posted))
            .map_err(|_| thread_exited())
    }

    fn sender(&self) -> Result<&SyncSender<Message<S>>, RhiError> {
        self.sender.as_ref().ok_or_else(thread_exited)
    }
}

impl<S: ThreadState> Drop for RenderThread<S> {
    /// Lets the thread handle every queued message, drop its state and exit, then joins it.
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                tracing::error!("the OpenGL render thread panicked during shutdown");
            }
        }
    }
}

/// The render thread's message loop. Returns when every sender is gone.
fn run<S: ThreadState>(mut state: S, receiver: &Receiver<Message<S>>) {
    loop {
        let message = match state.poll() {
            None => match receiver.recv() {
                Ok(message) => message,
                Err(_) => break,
            },
            Some(timeout) => match receiver.recv_timeout(timeout) {
                Ok(message) => message,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            },
        };
        match message {
            // The job catches panics from the caller's closure itself.
            Message::Call(job) => job.run(&mut state),
            Message::Post(posted) => {
                let outcome = panic::catch_unwind(AssertUnwindSafe(|| state.handle_posted(posted)));
                if let Err(payload) = outcome {
                    state.posted_panicked(panic_message(&*payload));
                }
            }
        }
    }
}

fn thread_exited() -> RhiError {
    RhiError::BackendError("the OpenGL render thread has exited".into())
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Records where and in which order work ran.
    struct Recorder {
        thread: ThreadId,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl ThreadState for Recorder {
        type Posted = String;

        fn handle_posted(&mut self, posted: String) {
            assert!(posted != "panic", "posted panic");
            self.log.lock().unwrap().push(posted);
        }

        fn posted_panicked(&mut self, message: String) {
            self.log
                .lock()
                .unwrap()
                .push(format!("panicked: {message}"));
        }
    }

    impl Drop for Recorder {
        fn drop(&mut self) {
            let on_owner = thread::current().id() == self.thread;
            self.log
                .lock()
                .unwrap()
                .push(format!("dropped on render thread: {on_owner}"));
        }
    }

    fn spawn() -> (RenderThread<Recorder>, Arc<Mutex<Vec<String>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let shared = Arc::clone(&log);
        let thread = RenderThread::spawn("test-render", move || {
            Ok(Recorder {
                thread: thread::current().id(),
                log: shared,
            })
        })
        .expect("spawn");
        (thread, log)
    }

    #[test]
    fn calls_run_on_the_render_thread_and_may_borrow() {
        let (render, _) = spawn();
        let data = [1_u8, 2, 3];
        let (thread_id, sum) = render
            .call(|state| {
                (
                    state.thread,
                    data.iter().map(|&b| u32::from(b)).sum::<u32>(),
                )
            })
            .expect("call");
        assert_ne!(thread_id, thread::current().id());
        assert_eq!(sum, 6);
        assert_eq!(render.call(|state| state.thread).expect("call"), thread_id);
    }

    #[test]
    fn calls_from_many_threads_all_complete() {
        let (render, _) = spawn();
        let render = Arc::new(render);
        let workers: Vec<_> = (0..8)
            .map(|i| {
                let render = Arc::clone(&render);
                thread::spawn(move || {
                    (0..200)
                        .map(|j| render.call(move |_| i * 1000 + j).expect("call"))
                        .sum::<u64>()
                })
            })
            .collect();
        let total: u64 = workers.into_iter().map(|w| w.join().unwrap()).sum();
        let expected: u64 = (0..8_u64)
            .map(|i| (0..200_u64).map(|j| i * 1000 + j).sum::<u64>())
            .sum();
        assert_eq!(total, expected);
    }

    #[test]
    fn posts_are_handled_in_order_before_later_calls() {
        let (render, log) = spawn();
        for i in 0..100 {
            render.post(format!("{i}")).expect("post");
        }
        let seen = render
            .call(|state| state.log.lock().unwrap().clone())
            .expect("call");
        let expected: Vec<String> = (0..100).map(|i| format!("{i}")).collect();
        assert_eq!(seen, expected);
        drop(render);
        assert_eq!(
            log.lock().unwrap().last().map(String::as_str),
            Some("dropped on render thread: true")
        );
    }

    #[test]
    fn a_panicking_call_resumes_on_the_caller_and_the_thread_survives() {
        let (render, _) = spawn();
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            render.call(|_| -> u32 { panic!("boom") })
        }));
        assert!(outcome.is_err(), "the panic must reach the caller");
        assert_eq!(render.call(|_| 7).expect("call after panic"), 7);
    }

    #[test]
    fn a_panicking_post_is_recorded_and_the_thread_survives() {
        let (render, log) = spawn();
        render.post("panic".into()).expect("post");
        render.post("after".into()).expect("post");
        render.call(|_| ()).expect("call");
        let log = log.lock().unwrap().clone();
        assert!(log[0].starts_with("panicked: posted panic"), "{log:?}");
        assert_eq!(log[1], "after");
    }

    #[test]
    fn initialization_errors_are_returned() {
        let result = RenderThread::<Recorder>::spawn("test-render", || {
            Err(RhiError::InitializationFailed("no context".into()))
        });
        assert!(matches!(result, Err(RhiError::InitializationFailed(m)) if m == "no context"));
    }

    #[test]
    fn calling_from_the_render_thread_is_an_error_not_a_deadlock() {
        let (render, _) = spawn();
        let render = Arc::new(render);
        let inner = Arc::clone(&render);
        let nested = render.call(move |_| inner.call(|_| ()).is_err());
        assert_eq!(nested.ok(), Some(true));
    }
}
