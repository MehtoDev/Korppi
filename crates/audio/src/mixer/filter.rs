//! A voice's low-pass filter: a two-pole state-variable filter per ear whose cutoff can move while
//! it runs (AUD-06), and the short delay lines that give each ear its own arrival time (AUD-05).
//!
//! The topology-preserving form (Simper's "SVF" from trapezoidal integration) stays stable and
//! free of zipper noise when its coefficients change between samples, unlike a direct-form
//! biquad, so the mixer recomputes them every [`SUB_BLOCK`] frames along the cutoff's ramp. The
//! cutoff ramps in octaves, so a sweep sounds even. With Q = 1/√2 the response is Butterworth:
//! -3 dB at the cutoff and 12 dB per octave above it. Each ear has its own cutoff (the head
//! shadows the far ear), and an ear whose cutoff is open passes its input unchanged.

use super::render::Ramp;

/// Frames between coefficient updates while the cutoff moves.
const SUB_BLOCK: usize = 16;
/// The lowest cutoff, in Hz.
const MIN_CUTOFF: f32 = 20.0;
/// 1/Q for a Butterworth response.
const DAMPING: f32 = std::f32::consts::SQRT_2;
/// Frames of each ear's delay line: 2.7 ms at 48 kHz, beyond the largest interaural delay
/// (0.66 ms) at rates up to 192 kHz.
const DELAY_FRAMES: usize = 128;
/// The longest delay read from the line, leaving room for the interpolation's neighbours.
const MAX_DELAY: f32 = (DELAY_FRAMES - 3) as f32;

/// A stereo low-pass filter with a ramping cutoff per ear.
#[derive(Clone, Debug)]
pub(super) struct LowPass {
    /// Each ear's cutoff in octaves above 1 Hz (`log2` of Hz).
    cutoff: [Ramp; 2],
    /// The cutoff at and above which an ear is open: bypassed once its ramp has settled.
    open: f32,
    a1: [f32; 2],
    a2: [f32; 2],
    a3: [f32; 2],
    /// Integrator states per ear.
    ic1: [f32; 2],
    ic2: [f32; 2],
    active: [bool; 2],
    /// Frames until the coefficients are next updated.
    countdown: usize,
}

impl LowPass {
    /// An open filter, with `open_hz` the cutoff that counts as open.
    pub(super) fn new(open_hz: f32) -> Self {
        let open = open_hz.max(MIN_CUTOFF).log2();
        Self {
            cutoff: [Ramp::at(open); 2],
            open,
            a1: [0.0; 2],
            a2: [0.0; 2],
            a3: [0.0; 2],
            ic1: [0.0; 2],
            ic2: [0.0; 2],
            active: [false; 2],
            countdown: 0,
        }
    }

    /// Moves each ear's cutoff to `hz` over `frames` frames.
    pub(super) fn set_cutoffs(&mut self, hz: [f32; 2], frames: u32) {
        for (cutoff, hz) in self.cutoff.iter_mut().zip(hz) {
            cutoff.to(hz.max(MIN_CUTOFF).log2().min(self.open), frames);
        }
    }

    /// Advances the cutoffs `frames` frames without filtering (a voice that is not mixed); the
    /// filter starts again from its input when it is next used.
    pub(super) fn skip(&mut self, frames: u32) {
        for cutoff in &mut self.cutoff {
            cutoff.skip(frames);
        }
        self.active = [false; 2];
        self.countdown = 0;
    }

    /// Prepares the next [`SUB_BLOCK`] frames at `device_rate`, given the voice's current
    /// input: an ear is bypassed while open, else gets coefficients for its cutoff now.
    fn update(&mut self, device_rate: f32, input: [f32; 2]) {
        for (ear, &input) in input.iter().enumerate() {
            let log2_hz = self.cutoff[ear].peek();
            if self.cutoff[ear].is_steady() && log2_hz >= self.open {
                self.active[ear] = false;
                continue;
            }
            if !self.active[ear] {
                // Start from the steady state of a constant input, so that engaging does not
                // click.
                self.ic1[ear] = 0.0;
                self.ic2[ear] = input;
                self.active[ear] = true;
            }
            let hz = log2_hz.exp2().min(0.45 * device_rate);
            let g = (std::f32::consts::PI * hz / device_rate).tan();
            self.a1[ear] = 1.0 / (1.0 + g * (g + DAMPING));
            self.a2[ear] = g * self.a1[ear];
            self.a3[ear] = g * self.a2[ear];
        }
    }

    /// Filters one frame at `device_rate`, advancing the cutoffs' ramps one frame.
    #[inline]
    pub(super) fn process(&mut self, device_rate: f32, input: [f32; 2]) -> [f32; 2] {
        if self.countdown == 0 {
            self.update(device_rate, input);
            self.countdown = SUB_BLOCK;
        }
        self.countdown -= 1;
        let mut output = input;
        for ear in 0..2 {
            self.cutoff[ear].next();
            if !self.active[ear] {
                continue;
            }
            let v3 = input[ear] - self.ic2[ear];
            let v1 = self.a1[ear] * self.ic1[ear] + self.a2[ear] * v3;
            let v2 = self.ic2[ear] + self.a2[ear] * self.ic1[ear] + self.a3[ear] * v3;
            self.ic1[ear] = 2.0 * v1 - self.ic1[ear];
            self.ic2[ear] = 2.0 * v2 - self.ic2[ear];
            output[ear] = v2;
        }
        output
    }
}

/// Each ear's arrival delay: a short line of the voice's recent frames, read at a fractional
/// delay (in frames) that ramps like a gain. At a delay of 0 an ear reads its input exactly.
#[derive(Clone, Debug)]
pub(super) struct EarDelay {
    line: [[f32; 2]; DELAY_FRAMES],
    /// Where the next frame goes.
    write: usize,
    delay: [Ramp; 2],
}

impl EarDelay {
    pub(super) fn new() -> Self {
        Self {
            line: [[0.0; 2]; DELAY_FRAMES],
            write: 0,
            delay: [Ramp::at(0.0); 2],
        }
    }

    /// Moves each ear's delay to `frames` frames, over `ramp` output frames.
    pub(super) fn set(&mut self, frames: [f32; 2], ramp: u32) {
        for (delay, frames) in self.delay.iter_mut().zip(frames) {
            delay.to(frames.clamp(0.0, MAX_DELAY), ramp);
        }
    }

    /// Advances the delays `frames` frames without delaying anything (a voice that is not
    /// mixed).
    pub(super) fn skip(&mut self, frames: u32) {
        for delay in &mut self.delay {
            delay.skip(frames);
        }
    }

    /// The frame `ago` frames before the newest one written, for `ear`.
    #[inline]
    fn past(&self, ago: usize, ear: usize) -> f32 {
        self.line[(self.write + DELAY_FRAMES - 1 - ago.min(DELAY_FRAMES - 1)) % DELAY_FRAMES][ear]
    }

    /// Takes one frame and returns each ear's frame from its delay ago, advancing the delays.
    #[inline]
    pub(super) fn process(&mut self, input: [f32; 2]) -> [f32; 2] {
        self.line[self.write] = input;
        self.write = (self.write + 1) % DELAY_FRAMES;
        let mut output = input;
        for (ear, sample) in output.iter_mut().enumerate() {
            let delay = self.delay[ear].next();
            if delay == 0.0 {
                continue;
            }
            // Between `whole` and `whole + 1` frames ago, with the newer and older neighbours.
            let whole = delay.floor();
            let t = delay - whole;
            let whole = whole as usize;
            let newer = if whole == 0 { 0 } else { whole - 1 };
            *sample = super::render::hermite(
                self.past(newer, ear),
                self.past(whole, ear),
                self.past(whole + 1, ear),
                self.past(whole + 2, ear),
                t,
            );
        }
        output
    }
}

/// One tap of a reflection slot: its delay in frames and its gain per ear, each ramping.
#[derive(Clone, Copy, Debug)]
struct Tap {
    delay: Ramp,
    gain: [Ramp; 2],
}

/// A voice's early reflections: a line of its recent mono signal (after its gain, before its
/// filter and panning) read by up to [`super::MAX_TAPS`] taps at fractional delays, each with a
/// gain per ear. The mixer keeps a fixed number of these, allocated when it is created, and
/// lends them to voices. Once its voice lets it go, a slot keeps sounding into its bus until
/// its line holds only silence, so the reflections of a voice's last sound are not cut off.
#[derive(Clone, Debug)]
pub(super) struct ReflectionSlot {
    line: Box<[f32]>,
    write: usize,
    taps: [Tap; super::MAX_TAPS],
    count: usize,
    /// Whether a voice holds it.
    pub(super) in_use: bool,
    /// The bus it sounds into.
    pub(super) bus: usize,
    /// Silent frames written in a row; the line is quiet once they fill it.
    silent_for: usize,
}

impl ReflectionSlot {
    /// A free slot whose line holds `frames` frames.
    pub(super) fn new(frames: usize) -> Self {
        let frames = frames.max(4);
        Self {
            line: vec![0.0; frames].into_boxed_slice(),
            write: 0,
            taps: [Tap {
                delay: Ramp::at(0.0),
                gain: [Ramp::at(0.0); 2],
            }; super::MAX_TAPS],
            count: 0,
            in_use: false,
            bus: 0,
            silent_for: frames,
        }
    }

    /// Whether its line holds only silence: nothing more comes out until it is fed.
    pub(super) fn is_quiet(&self) -> bool {
        self.silent_for >= self.line.len()
    }

    /// Takes the slot for a voice, from silence and without taps.
    pub(super) fn assign(&mut self) {
        if !self.is_quiet() {
            self.line.fill(0.0);
            self.silent_for = self.line.len();
        }
        self.in_use = true;
        self.count = 0;
    }

    /// Lets the slot go: it sounds out what its line holds, then is quiet.
    pub(super) fn release(&mut self) {
        self.in_use = false;
    }

    /// Moves the taps to `reflections` (delays converted at `rate` frames per second) over
    /// `frames` frames; taps new to the slot start where they are going.
    pub(super) fn set(&mut self, reflections: &super::Reflections, rate: f32, frames: u32) {
        let longest = (self.line.len() - 3) as f32;
        let count = reflections.count.min(super::MAX_TAPS);
        for (index, (tap, target)) in self
            .taps
            .iter_mut()
            .zip(&reflections.taps[..count])
            .enumerate()
        {
            let delay = (target.delay * rate).clamp(0.0, longest);
            let gains = [target.gain_left.max(0.0), target.gain_right.max(0.0)];
            let ramp = if index < self.count { frames } else { 0 };
            tap.delay.to(delay, ramp);
            tap.gain[0].to(gains[0], ramp);
            tap.gain[1].to(gains[1], ramp);
        }
        self.count = count;
    }

    /// Takes one mono frame and returns the taps' sum per ear, advancing their ramps.
    #[inline]
    pub(super) fn process(&mut self, input: f32) -> [f32; 2] {
        self.line[self.write] = input;
        self.write = (self.write + 1) % self.line.len();
        self.silent_for = if input == 0.0 {
            self.silent_for.saturating_add(1)
        } else {
            0
        };
        let (line, newest) = (&self.line, self.write + self.line.len() - 1);
        // The sample `ago` frames before the newest written.
        let past = |ago: usize| line[(newest - ago.min(line.len() - 1)) % line.len()];
        let mut output = [0.0; 2];
        for tap in &mut self.taps[..self.count] {
            let delay = tap.delay.next();
            let gains = [tap.gain[0].next(), tap.gain[1].next()];
            let whole = delay.floor();
            let t = delay - whole;
            let whole = whole as usize;
            let value = super::render::hermite(
                past(whole.saturating_sub(1)),
                past(whole),
                past(whole + 1),
                past(whole + 2),
                t,
            );
            output[0] += value * gains[0];
            output[1] += value * gains[1];
        }
        output
    }

    /// Feeds silence for `out.len() / 2` frames, adding the taps to the interleaved stereo `out`.
    pub(super) fn sound_out(&mut self, out: &mut [f32]) {
        for frame in out.as_chunks_mut::<2>().0 {
            if self.is_quiet() {
                break;
            }
            let [left, right] = self.process(0.0);
            frame[0] += left;
            frame[1] += right;
        }
    }
}
