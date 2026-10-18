use std::sync::Mutex;

/// Environment variable that overrides whether API-level validation is enabled.
///
/// `1`/`true`/`on` enables it (also in release builds); `0`/`false`/`off` disables it (also in
/// debug builds). When unset, validation follows `cfg!(debug_assertions)`.
pub const VALIDATION_ENV_VAR: &str = "KORPPI_RHI_VALIDATION";

/// Severity of a message reported by a backend's API-level debug machinery.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticSeverity {
    /// Informational message.
    Info,
    /// Suspicious or slow usage that is still valid.
    Warning,
    /// Invalid API usage or an API error.
    Error,
}

/// A message reported by the graphics API's debug machinery (Vulkan validation layers through
/// `VK_EXT_debug_utils`, OpenGL `KHR_debug` output).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendDiagnostic {
    /// Message severity.
    pub severity: DiagnosticSeverity,
    /// Message text as reported by the API or layer.
    pub message: String,
}

/// Thread-safe, bounded log that backends append diagnostics to from API callbacks.
///
/// API callbacks can run on any thread and must never panic across the FFI boundary, so every
/// method recovers from a poisoned lock instead of propagating it. At most
/// [`DiagnosticLog::CAPACITY`] entries are kept between two [`DiagnosticLog::take`] calls, so an
/// application that never drains the log does not grow without bound.
#[derive(Debug, Default)]
pub struct DiagnosticLog {
    state: Mutex<LogState>,
}

#[derive(Debug, Default)]
struct LogState {
    entries: Vec<BackendDiagnostic>,
    dropped: usize,
}

impl DiagnosticLog {
    /// Maximum number of entries kept between two calls to [`DiagnosticLog::take`].
    pub const CAPACITY: usize = 1024;

    /// Creates an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a diagnostic, or counts it as dropped once the log is full.
    pub fn push(&self, severity: DiagnosticSeverity, message: impl Into<String>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.entries.len() < Self::CAPACITY {
            state.entries.push(BackendDiagnostic {
                severity,
                message: message.into(),
            });
        } else {
            state.dropped += 1;
        }
    }

    /// Removes and returns every diagnostic recorded so far, oldest first. If entries were
    /// dropped because the log was full, a final warning says how many.
    pub fn take(&self) -> Vec<BackendDiagnostic> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entries = std::mem::take(&mut state.entries);
        let dropped = std::mem::take(&mut state.dropped);
        if dropped > 0 {
            entries.push(BackendDiagnostic {
                severity: DiagnosticSeverity::Warning,
                message: format!(
                    "{dropped} further diagnostics were dropped because the log was full"
                ),
            });
        }
        entries
    }
}

/// Decides whether API-level validation should be enabled, given the value of
/// [`VALIDATION_ENV_VAR`] (if set) and whether this is a debug build.
pub fn validation_requested(env_value: Option<&str>, debug_build: bool) -> bool {
    match env_value.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if matches!(v.as_str(), "1" | "true" | "on") => true,
        Some(v) if matches!(v.as_str(), "0" | "false" | "off") => false,
        _ => debug_build,
    }
}

/// Whether backends should enable API-level validation for the current process, from
/// [`VALIDATION_ENV_VAR`] and `cfg!(debug_assertions)`.
pub fn api_validation_requested() -> bool {
    validation_requested(
        std::env::var(VALIDATION_ENV_VAR).ok().as_deref(),
        cfg!(debug_assertions),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_follows_build_profile_when_unset() {
        assert!(validation_requested(None, true));
        assert!(!validation_requested(None, false));
    }

    #[test]
    fn environment_overrides_build_profile() {
        for on in ["1", "true", "ON", " on "] {
            assert!(validation_requested(Some(on), false), "{on:?}");
        }
        for off in ["0", "false", "Off"] {
            assert!(!validation_requested(Some(off), true), "{off:?}");
        }
        assert!(validation_requested(Some("unexpected"), true));
        assert!(!validation_requested(Some("unexpected"), false));
    }

    #[test]
    fn log_take_drains_in_order() {
        let log = DiagnosticLog::new();
        log.push(DiagnosticSeverity::Warning, "first");
        log.push(DiagnosticSeverity::Error, String::from("second"));
        let taken = log.take();
        assert_eq!(
            taken,
            vec![
                BackendDiagnostic {
                    severity: DiagnosticSeverity::Warning,
                    message: "first".into()
                },
                BackendDiagnostic {
                    severity: DiagnosticSeverity::Error,
                    message: "second".into()
                },
            ]
        );
        assert!(log.take().is_empty());
    }

    #[test]
    fn log_is_bounded_and_reports_dropped_entries() {
        let log = DiagnosticLog::new();
        for i in 0..DiagnosticLog::CAPACITY + 10 {
            log.push(DiagnosticSeverity::Warning, format!("message {i}"));
        }
        let taken = log.take();
        assert_eq!(taken.len(), DiagnosticLog::CAPACITY + 1);
        let summary = taken.last().expect("summary entry");
        assert_eq!(summary.severity, DiagnosticSeverity::Warning);
        assert!(summary
            .message
            .contains("10 further diagnostics were dropped"));
        // The drop counter resets with each take.
        log.push(DiagnosticSeverity::Error, "fresh");
        assert_eq!(log.take().len(), 1);
    }

    #[test]
    fn log_recovers_from_poisoned_lock() {
        let log = std::sync::Arc::new(DiagnosticLog::new());
        let poisoner = std::sync::Arc::clone(&log);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.state.lock().unwrap();
            panic!("poison the log");
        })
        .join();
        log.push(DiagnosticSeverity::Error, "after poison");
        assert_eq!(log.take().len(), 1);
    }
}
