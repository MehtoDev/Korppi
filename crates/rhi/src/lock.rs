//! Mutex locking that survives poisoning (finding RVK-14).

use std::sync::{Mutex, MutexGuard};

/// Locks `mutex`, taking the guard even if a thread panicked while holding it.
///
/// The backends' mutexes guard registries and caches whose entries stay valid across a panic
/// (each update is a single insert, remove or field write), so one failing caller must not turn
/// every later device call into a panic.
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_poisoned_mutex_still_locks() {
        let mutex = std::sync::Arc::new(Mutex::new(1));
        let poisoner = std::sync::Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().unwrap();
            panic!("poison the mutex");
        })
        .join();
        assert!(mutex.is_poisoned());
        *lock(&mutex) += 1;
        assert_eq!(*lock(&mutex), 2);
    }
}
