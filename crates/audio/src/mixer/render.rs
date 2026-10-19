//! A voice on the audio thread: its ramps and fades, and how it renders into a bus.

use super::clip::ClipData;
use super::control::{Fade, FadeCurve, VoiceHandle};
use super::filter::{EarDelay, LowPass, ReflectionSlot};

/// A value moving linearly to a target, one step per output frame.
#[derive(Copy, Clone, Debug)]
pub(super) struct Ramp {
    value: f32,
    target: f32,
    step: f32,
    remaining: u32,
}

impl Ramp {
    pub(super) const fn at(value: f32) -> Self {
        Self {
            value,
            target: value,
            step: 0.0,
            remaining: 0,
        }
    }

    /// Moves to `target` over `frames` frames.
    pub(super) fn to(&mut self, target: f32, frames: u32) {
        if frames == 0 {
            *self = Self::at(target);
        } else {
            self.target = target;
            self.step = (target - self.value) / frames as f32;
            self.remaining = frames;
        }
    }

    /// The value for this frame; then advances one frame. Each value is computed from the
    /// target and the frames left, so rounding does not build up over a long ramp.
    #[inline]
    pub(super) fn next(&mut self) -> f32 {
        let value = self.value;
        if self.remaining > 0 {
            self.remaining -= 1;
            self.value = self.target - self.step * self.remaining as f32;
        }
        value
    }

    /// The value for this frame, without advancing.
    pub(super) fn peek(&self) -> f32 {
        self.value
    }

    /// Whether the value has reached its target.
    pub(super) fn is_steady(&self) -> bool {
        self.remaining == 0
    }

    /// Advances `frames` frames at once.
    pub(super) fn skip(&mut self, frames: u32) {
        if frames >= self.remaining {
            *self = Self::at(self.target);
        } else {
            self.remaining -= frames;
            self.value = self.target - self.step * self.remaining as f32;
        }
    }

    /// The larger of the value and the target: how loud a ramping gain counts for ranking, so
    /// that a fade-down stays audible until it ends and a fade-up is audible from its start.
    pub(super) fn peak(&self) -> f32 {
        self.value.max(self.target)
    }

    fn is_steady_at(&self, value: f32) -> bool {
        self.remaining == 0 && self.value == value
    }
}

/// A voice's fade state.
#[derive(Copy, Clone, Debug)]
pub(super) enum Envelope {
    /// Full gain.
    Full,
    /// Fading in: `position` frames of `frames` done.
    In {
        position: u32,
        frames: u32,
        curve: FadeCurve,
    },
    /// Fading out from gain `from`; the voice ends when `position` reaches `frames`.
    Out {
        position: u32,
        frames: u32,
        curve: FadeCurve,
        from: f32,
    },
}

impl Envelope {
    pub(super) fn fade_in(fade: Fade) -> Self {
        if fade.frames == 0 {
            Envelope::Full
        } else {
            Envelope::In {
                position: 0,
                frames: fade.frames,
                curve: fade.curve,
            }
        }
    }

    fn level(curve: FadeCurve, x: f32) -> f32 {
        match curve {
            FadeCurve::Linear => x,
            FadeCurve::EqualPower => (x * std::f32::consts::FRAC_PI_2).sin(),
        }
    }

    /// The gain for this frame, then advances; `None` once a fade-out has ended.
    #[inline]
    fn next(&mut self) -> Option<f32> {
        match *self {
            Envelope::Full => Some(1.0),
            Envelope::In {
                position,
                frames,
                curve,
            } => {
                let gain = Self::level(curve, position as f32 / frames as f32);
                *self = if position + 1 >= frames {
                    Envelope::Full
                } else {
                    Envelope::In {
                        position: position + 1,
                        frames,
                        curve,
                    }
                };
                Some(gain)
            }
            Envelope::Out {
                position,
                frames,
                curve,
                from,
            } => {
                if position >= frames {
                    return None;
                }
                let gain = from * Self::level(curve, 1.0 - position as f32 / frames as f32);
                *self = Envelope::Out {
                    position: position + 1,
                    frames,
                    curve,
                    from,
                };
                Some(gain)
            }
        }
    }

    /// The gain the envelope is at now, without advancing.
    fn current(&self) -> f32 {
        let mut copy = *self;
        copy.next().unwrap_or(0.0)
    }

    /// How loud the envelope counts for ranking: a fade-in counts as full, a fade-out as where
    /// it is.
    pub(super) fn peak(&self) -> f32 {
        match self {
            Envelope::Out { .. } => self.current(),
            _ => 1.0,
        }
    }

    fn has_ended(&self) -> bool {
        matches!(*self, Envelope::Out { position, frames, .. } if position >= frames)
    }
}

/// A playing voice.
pub(super) struct Voice {
    pub(super) handle: VoiceHandle,
    pub(super) clip: usize,
    /// Position in the clip, in clip frames.
    pub(super) cursor: f64,
    pub(super) gain: Ramp,
    pub(super) pitch: Ramp,
    /// Left and right gains (spatialization).
    pub(super) pan: [Ramp; 2],
    pub(super) filter: LowPass,
    /// Each ear's arrival delay (the interaural time difference).
    pub(super) delay: EarDelay,
    /// The bus the voice sends to (`usize::MAX` for none), and the send's level.
    pub(super) send_bus: usize,
    pub(super) send: Ramp,
    /// The mixer's reflection line lent to the voice (`usize::MAX` for none).
    pub(super) reflection: usize,
    pub(super) looping: bool,
    /// The loop region, in clip frames: a looping voice plays up to `loop_end`, then returns to
    /// `loop_start`.
    pub(super) loop_start: f64,
    pub(super) loop_end: f64,
    /// Whether the voice has looped at least once, so that interpolation before `loop_start`
    /// reads the end of the region instead of the intro.
    pub(super) wrapped: bool,
    pub(super) bus: usize,
    pub(super) priority: u8,
    /// The output frame the voice starts sounding at.
    pub(super) start_at: u64,
    /// A scheduled stop: the output frame and the fade-out.
    pub(super) stop_at: Option<(u64, Fade)>,
    pub(super) envelope: Envelope,
    /// Whether the voice is among those mixed (else it is virtual).
    pub(super) audible: bool,
    /// 1 while audible, 0 while virtual; ramps between them.
    pub(super) presence: Ramp,
    /// Whether the voice has not rendered a frame yet, so that it can become audible or virtual
    /// at once instead of fading, and takes new gains, pitch and spatial settings at once.
    pub(super) fresh: bool,
    /// The output frame the voice first rendered.
    pub(super) started: Option<u64>,
    /// Whether the voice is paused or pausing.
    pub(super) paused: bool,
    /// 1 while running, 0 while paused; ramps between them. A paused voice holds its place once
    /// this reaches 0.
    pub(super) run: Ramp,
    /// Its audibility for this block's ranking.
    pub(super) rank_level: f32,
}

impl Voice {
    /// Makes the voice audible or virtual, fading over `min_ramp` frames once it has sounded.
    pub(super) fn set_audible(&mut self, audible: bool, min_ramp: u32) {
        if audible == self.audible {
            return;
        }
        self.audible = audible;
        let target = if audible { 1.0 } else { 0.0 };
        if self.fresh {
            self.presence = Ramp::at(target);
        } else {
            self.presence.to(target, min_ramp);
        }
    }

    /// Whether the voice is in the mix now (audible, or fading to virtual).
    pub(super) fn is_sounding(&self) -> bool {
        !self.fresh && self.presence.peak() > 0.0
    }

    /// Whether the voice is paused and silent: it holds its place and needs no mixing.
    pub(super) fn is_held(&self) -> bool {
        self.paused && self.run.is_steady() && self.run.peek() == 0.0
    }

    /// Stops the voice at its next frame, fading out over `min_ramp` frames.
    pub(super) fn stop_now(&mut self) {
        self.stop_at = Some((0, Fade::NONE));
    }

    /// The frame where playback ends or wraps.
    fn end(&self, frames: f64) -> f64 {
        if self.looping {
            self.loop_end
        } else {
            frames
        }
    }

    /// Moves a cursor past the loop's end back into the region, keeping its overshoot.
    fn wrap(&mut self) {
        let length = self.loop_end - self.loop_start;
        self.cursor = self.loop_start + (self.cursor - self.loop_start) % length;
        self.wrapped = true;
    }

    /// Renders the voice into `mix` (interleaved stereo) for the block starting at output frame
    /// `block_start`, at `rate` clip frames per output frame times its pitch. A virtual voice
    /// only advances, and a held one stays where it is. True once the voice has ended.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render(
        &mut self,
        clip: &ClipData,
        mix: &mut [f32],
        mut send: Option<&mut [f32]>,
        mut reflections: Option<&mut ReflectionSlot>,
        block_start: u64,
        rate: f64,
        device_rate: f32,
        min_ramp: u32,
    ) -> bool {
        let frames = clip.frames() as f64;
        if frames == 0.0 {
            return true;
        }
        let block = mix.len() / 2;
        // A virtual voice skips ahead once its reflections have sounded out.
        if self.is_idle_for(block_start, block as u64)
            && reflections.as_deref().is_none_or(ReflectionSlot::is_quiet)
        {
            return self.advance(block_start, block, frames, rate);
        }
        let mut ended = None;
        for (offset, out) in mix.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let now = block_start + offset as u64;
            if now < self.start_at {
                continue;
            }
            if self.is_held() {
                if let Some((at, _)) = self.stop_at.as_mut() {
                    if *at <= block_start {
                        ended = Some(offset);
                        break;
                    }
                    // Shift future scheduled stop by the remaining held frames in this block.
                    let held_frames = (block - offset) as u64;
                    if reflections.as_deref().is_none_or(ReflectionSlot::is_quiet) {
                        *at = at.saturating_add(held_frames);
                        break;
                    }
                    *at = at.saturating_add(1);
                }
                // Held, the voice is silent; its reflections still sound out.
                match reflections.as_deref_mut() {
                    Some(slot) if !slot.is_quiet() => slot.sound_out(out),
                    _ => break,
                }
                continue;
            }
            if self.fresh {
                self.started = Some(now);
                self.fresh = false;
            }
            if let Some((at, fade)) = self.stop_at {
                if now >= at && !matches!(self.envelope, Envelope::Out { .. }) {
                    self.envelope = Envelope::Out {
                        position: 0,
                        // An empty fade still declicks over the minimum.
                        frames: if fade.frames == 0 {
                            min_ramp
                        } else {
                            fade.frames
                        },
                        curve: fade.curve,
                        from: self.envelope.current(),
                    };
                }
            }
            let Some(envelope) = self.envelope.next() else {
                ended = Some(offset);
                break;
            };
            if self.cursor >= self.end(frames) {
                if self.looping {
                    self.wrap();
                } else {
                    ended = Some(offset);
                    break;
                }
            }
            let presence = self.presence.next();
            let gain = self.gain.next() * envelope * presence * self.run.next();
            let pan = [self.pan[0].next(), self.pan[1].next()];
            let send_level = self.send.next();
            if presence > 0.0 {
                let input = [self.sample(clip, 0), self.sample(clip, 1)];
                if let Some(slot) = reflections.as_deref_mut() {
                    // After the gain, before the filter, the ears' delays and the panning: the
                    // reflections arrive from their own directions.
                    let [left, right] = slot.process((input[0] + input[1]) * 0.5 * gain);
                    out[0] += left;
                    out[1] += right;
                }
                let filtered = self.filter.process(device_rate, input);
                if let Some(send) = send.as_deref_mut() {
                    // After the gain and filter, before the ears' delays and the panning.
                    send[offset] += (filtered[0] + filtered[1]) * 0.5 * gain * send_level;
                }
                let [left, right] = self.delay.process(filtered);
                out[0] += left * gain * pan[0];
                out[1] += right * gain * pan[1];
            } else {
                self.filter.skip(1);
                self.delay.skip(1);
                if let Some(slot) = reflections.as_deref_mut() {
                    let [left, right] = slot.process(0.0);
                    out[0] += left;
                    out[1] += right;
                }
            }
            self.cursor += f64::from(self.pitch.next()) * rate;
        }
        if let Some(offset) = ended {
            // The rest of the block, the reflections sound out (then the mixer lets their line
            // go, and it carries on from the next block).
            if let Some(slot) = reflections {
                slot.sound_out(&mut mix[2 * offset..]);
            }
            return true;
        }
        (!self.looping && self.cursor >= frames) || self.envelope.has_ended()
    }

    /// Whether, for the block of `block` frames at `block_start`, the voice is virtual and
    /// nothing about it changes but its gains and cutoff: it can advance without rendering
    /// frame by frame.
    fn is_idle_for(&self, block_start: u64, block: u64) -> bool {
        !self.audible
            && self.presence.is_steady_at(0.0)
            && self.pitch.remaining == 0
            && !self.paused
            && self.run.is_steady_at(1.0)
            && matches!(self.envelope, Envelope::Full)
            && self.start_at <= block_start
            && self.stop_at.is_none_or(|(at, _)| at >= block_start + block)
    }

    /// Advances an idle voice `block` frames at once; true if it reached the end of its clip.
    fn advance(&mut self, block_start: u64, block: usize, frames: f64, rate: f64) -> bool {
        if self.fresh {
            self.started = Some(block_start);
            self.fresh = false;
        }
        self.gain.skip(block as u32);
        self.pan[0].skip(block as u32);
        self.pan[1].skip(block as u32);
        self.send.skip(block as u32);
        self.filter.skip(block as u32);
        self.delay.skip(block as u32);
        self.cursor += f64::from(self.pitch.value) * rate * block as f64;
        if self.cursor >= self.end(frames) {
            if !self.looping {
                return true;
            }
            self.wrap();
        }
        false
    }

    /// Channel `channel` of the clip at the cursor, by cubic interpolation. A looping voice's
    /// neighbours past the loop's end come from its start, and once it has looped, those before
    /// its start from its end.
    #[inline]
    fn sample(&self, clip: &ClipData, channel: usize) -> f32 {
        let base = self.cursor.floor();
        let t = (self.cursor - base) as f32;
        let base = base as isize;
        let (start, end) = (self.loop_start as isize, self.loop_end as isize);
        let at = |offset: isize| {
            let mut frame = base + offset;
            if self.looping && (frame >= end || (self.wrapped && frame < start)) {
                frame = start + (frame - start).rem_euclid(end - start);
            }
            clip.sample(frame, channel)
        };
        hermite(at(-1), at(0), at(1), at(2), t)
    }
}

/// Cubic Hermite interpolation between `x1` and `x2` at `t` in `[0, 1)`; exactly `x1` at 0.
#[inline]
pub(super) fn hermite(x0: f32, x1: f32, x2: f32, x3: f32, t: f32) -> f32 {
    let c1 = 0.5 * (x2 - x0);
    let c2 = x0 - 2.5 * x1 + 2.0 * x2 - 0.5 * x3;
    let c3 = 0.5 * (x3 - x0) + 1.5 * (x1 - x2);
    ((c3 * t + c2) * t + c1) * t + x1
}
