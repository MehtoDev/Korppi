//! Queue submissions and frames in flight (finding RHI-02).
//!
//! `GraphicsDevice::submit` returns as soon as the work is queued and hands back a
//! [`SubmissionId`]. Submissions on one device complete in the order they were made, so waiting
//! for an ID also waits for every earlier submission. `GraphicsDevice::begin_frame` bounds how
//! far the CPU runs ahead: it waits for the work of the frame that last used the same
//! [`FrameIndex`], [`FRAMES_IN_FLIGHT`] frames earlier, before that frame's transient resources
//! are reused.

use std::sync::Mutex;

/// Number of frames the CPU may record while the GPU still executes earlier ones.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Identifies one queue submission of a device.
///
/// IDs are strictly increasing in submission order (not necessarily consecutive: backends may
/// submit internal work such as uploads between them) and submissions complete in that order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubmissionId(u64);

impl SubmissionId {
    /// Wraps a raw ID. Backends start at 1.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw ID.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Slot of a frame in flight, in `0..FRAMES_IN_FLIGHT`. Index per-frame resources (uniform
/// rings, staging buffers) with it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameIndex(usize);

impl FrameIndex {
    /// Creates a frame index.
    ///
    /// # Panics
    /// Panics if `index >= FRAMES_IN_FLIGHT`.
    pub const fn new(index: usize) -> Self {
        assert!(index < FRAMES_IN_FLIGHT, "frame index out of range");
        Self(index)
    }

    /// Returns the slot as an array index.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Frames-in-flight bookkeeping shared by the backends: which submission each frame slot must
/// wait for before it is reused.
///
/// Every submission is attributed to the frame current when it was made (submissions before the
/// first `begin_frame` belong to frame 0).
#[derive(Debug, Default)]
pub struct FrameTracker {
    state: Mutex<FrameTrackerState>,
}

#[derive(Debug, Default)]
struct FrameTrackerState {
    /// Number of the current frame; `None` before the first `begin_frame`.
    frame: Option<u64>,
    /// Last submission made in each slot's current (or most recent) frame.
    last_submission: [Option<SubmissionId>; FRAMES_IN_FLIGHT],
}

impl FrameTrackerState {
    fn slot(&self) -> usize {
        (self.frame.unwrap_or(0) % FRAMES_IN_FLIGHT as u64) as usize
    }
}

impl FrameTracker {
    /// Creates a tracker in front of frame 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `submission` as part of the current frame.
    pub fn record(&self, submission: SubmissionId) {
        let mut state = self.lock();
        let slot = state.slot();
        let last = &mut state.last_submission[slot];
        *last = (*last).max(Some(submission));
    }

    /// Starts the next frame. Returns its slot and the last submission of the frame that used
    /// the slot before, which must complete before the slot's resources are reused.
    pub fn begin_frame(&self) -> (FrameIndex, Option<SubmissionId>) {
        let mut state = self.lock();
        state.frame = Some(state.frame.map_or(0, |frame| frame + 1));
        let slot = state.slot();
        let wait_for = state.last_submission[slot].take();
        (FrameIndex(slot), wait_for)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FrameTrackerState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_cycle_through_the_slots() {
        let tracker = FrameTracker::new();
        let slots: Vec<usize> = (0..5).map(|_| tracker.begin_frame().0.index()).collect();
        assert_eq!(slots, [0, 1, 0, 1, 0]);
    }

    #[test]
    fn a_slot_waits_for_the_last_submission_of_its_previous_frame() {
        let tracker = FrameTracker::new();
        assert_eq!(tracker.begin_frame(), (FrameIndex::new(0), None));
        tracker.record(SubmissionId::new(1));
        tracker.record(SubmissionId::new(3));
        assert_eq!(tracker.begin_frame(), (FrameIndex::new(1), None));
        tracker.record(SubmissionId::new(4));
        assert_eq!(
            tracker.begin_frame(),
            (FrameIndex::new(0), Some(SubmissionId::new(3)))
        );
        // Frame 2 made no submission, so reusing slot 0 again waits for nothing new.
        assert_eq!(
            tracker.begin_frame(),
            (FrameIndex::new(1), Some(SubmissionId::new(4)))
        );
        assert_eq!(tracker.begin_frame(), (FrameIndex::new(0), None));
    }

    #[test]
    fn submissions_before_the_first_frame_belong_to_frame_0() {
        let tracker = FrameTracker::new();
        tracker.record(SubmissionId::new(7));
        assert_eq!(
            tracker.begin_frame(),
            (FrameIndex::new(0), Some(SubmissionId::new(7)))
        );
    }

    #[test]
    #[should_panic(expected = "frame index out of range")]
    fn frame_indexes_are_bounded() {
        let _ = FrameIndex::new(FRAMES_IN_FLIGHT);
    }
}
