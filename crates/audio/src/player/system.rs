//! The per-tick update of every [`AudioPlayer`]: requests to voices, playlist scheduling,
//! spatialization and state.

use super::{
    AudioOcclusion, AudioPlayer, AudioRoom, AudioSource, PlayerState, PlayerVoice, Request,
    SpatialSettings,
};
use crate::ecs::{AudioListener, AudioSettings};
use crate::error::AudioResult;
use crate::mixer::{
    At, AudioVoices, BusEffect, BusId, Fade, ReflectionTap, Reflections, RegisteredClip,
    RoomReverb, VoiceHandle, VoiceParams, VoiceSpatial, MASTER,
};
use crate::raytrace::{
    AcousticGeometry, AcousticRay, EmitterReflections, RoomInput, RoomOutput, RoomWorker,
    MAX_RECEIVERS,
};
use crate::spatial::{
    AcousticRaycaster, EmitterState, EqualPowerPanner, ListenerState, RaytracedSpatializer,
    SpatialAudioModel, SpatializerMode, StandardSpatializer, VoiceSpatialParams,
};
use korppi_asset::AssetServer;
use korppi_ecs::{Commands, Entity, Local, Query, Res, ResMut};
use korppi_math::Vec3;
use korppi_physics::RigidBody;
use korppi_scene::GlobalTransform;
use std::sync::Arc;

/// How far ahead of a playlist transition the update schedules it, in seconds: more than a game
/// tick and a device period, so that the mixer receives it before its frame.
const SCHEDULE_AHEAD: f64 = 0.5;

/// What [`update_audio_players`] keeps between ticks.
pub struct PlayerSystemState {
    standard: StandardSpatializer,
    raytraced: RaytracedSpatializer,
    /// The master gain last sent, if any.
    master: Option<f32>,
    last_clock: u64,
    reverb: Option<ReverbBus>,
    /// The acoustics worker, in raytraced mode with a room geometry.
    worker: Option<RoomWorker>,
}

/// The "Reverb" bus spatialized players send to, and what was last set on it.
struct ReverbBus {
    bus: BusId,
    mode: SpatializerMode,
    room: RoomReverb,
    wet: f32,
    /// The decay time last given to the late reverb.
    rt60: Option<[f32; 3]>,
}

/// Late-reverb decay times for raytraced mode, until the acoustics worker reports a room's.
const DEFAULT_RT60: [f32; 3] = [1.0, 0.9, 0.7];

/// Keeps the "Reverb" bus in step with `settings`: Freeverb's room at the room's level in
/// standard mode, the late reverb at full level in raytraced mode. Returns the bus.
fn reverb_bus(
    state: &mut Option<ReverbBus>,
    voices: &mut AudioVoices,
    settings: &AudioSettings,
) -> Option<BusId> {
    let rate = voices.sample_rate();
    let mode = settings.spatializer_mode;
    let room = RoomReverb {
        size: settings.room_reverb.size,
        damping: settings.room_reverb.damping,
    };
    let wet = match mode {
        SpatializerMode::Standard => settings.room_reverb.wet.max(0.0),
        SpatializerMode::Raytraced => 1.0,
    };
    let effect = |mode| match mode {
        SpatializerMode::Standard => BusEffect::room(room, rate),
        SpatializerMode::Raytraced => BusEffect::late(DEFAULT_RT60, rate),
    };
    let control = voices.control();
    let Some(current) = state else {
        let bus = match control.bus("Reverb") {
            Some(bus) => bus,
            None => control.add_bus("Reverb", MASTER).ok()?,
        };
        sent(control.set_bus_effect(bus, Some(effect(mode))));
        sent(control.set_bus_gain(bus, wet));
        *state = Some(ReverbBus {
            bus,
            mode,
            room,
            wet,
            rt60: None,
        });
        return Some(bus);
    };
    if current.mode != mode {
        sent(control.set_bus_effect(current.bus, Some(effect(mode))));
        current.mode = mode;
        current.room = room;
        current.rt60 = None;
    } else if current.room != room {
        sent(control.set_room(current.bus, room));
        current.room = room;
    }
    if current.wet != wet {
        sent(control.set_bus_gain(current.bus, wet));
        current.wet = wet;
    }
    Some(current.bus)
}

impl Default for PlayerSystemState {
    fn default() -> Self {
        Self {
            standard: StandardSpatializer::new(48_000.0),
            raytraced: RaytracedSpatializer::new(48_000.0),
            master: None,
            last_clock: 0,
            reverb: None,
            worker: None,
        }
    }
}

/// What one player's update needs.
struct Context<'a> {
    voices: &'a mut AudioVoices,
    assets: &'a AssetServer,
    /// The mixer's clock at this update.
    clock: u64,
    /// Output frames per second.
    rate: f64,
    /// The ramp for spatial changes: about a tick, in output frames.
    tick_frames: u32,
    /// The player's stereo gains, filter cutoff and Doppler factor, if it is spatialized.
    spatial: Option<(VoiceSpatial, f32)>,
    /// The reverb bus and the player's send to it, if it is spatialized.
    reverb: Option<(BusId, f32)>,
    /// The player's early reflections, in raytraced mode once the worker has traced it.
    reflections: Option<Reflections>,
}

impl Context<'_> {
    fn clip(
        &mut self,
        clip: &korppi_asset::AssetHandle<crate::AudioClip>,
    ) -> Option<RegisteredClip> {
        match self.voices.clip(self.assets, clip) {
            Ok(clip) => clip,
            Err(error) => {
                tracing::warn!("cannot register an audio clip: {error}");
                None
            }
        }
    }

    fn frames(&self, seconds: f32) -> u32 {
        (f64::from(seconds) * self.rate).round() as u32
    }
}

/// Logs a command the mixer's full queue refused.
fn sent(result: AudioResult<()>) {
    if let Err(error) = result {
        tracing::warn!("an audio player's command was dropped: {error}");
    }
}

/// Updates every [`AudioPlayer`] once per game tick (see the [module documentation](super)):
/// applies their requests, schedules playlist transitions, spatializes players with a
/// `GlobalTransform` against the primary [`AudioListener`] (attenuated and muffled where
/// [`AudioOcclusion`] blocks the line of sight), sets the master volume from [`AudioSettings`]
/// and the listener, and writes each player's state back. Run it after transforms propagate.
#[allow(clippy::needless_pass_by_value, clippy::type_complexity)]
pub fn update_audio_players(
    mut players: Query<(
        Entity,
        &mut AudioPlayer,
        Option<&GlobalTransform>,
        Option<&RigidBody>,
    )>,
    mut listeners: Query<(&AudioListener, &GlobalTransform, Option<&RigidBody>)>,
    mut voices: ResMut<AudioVoices>,
    (assets, settings, occlusion): (Res<AssetServer>, Res<AudioSettings>, Res<AudioOcclusion>),
    mut room: ResMut<AudioRoom>,
    mut state: Local<PlayerSystemState>,
    mut commands: Commands,
) {
    voices.update();
    let clock = voices.clock().frames();
    let tick_frames = clock.saturating_sub(state.last_clock).clamp(240, 4800) as u32;
    state.last_clock = clock;

    let mut listener = ListenerState::default();
    let mut listener_volume = 1.0;
    for (ears, transform, body) in listeners.iter_mut() {
        if ears.active && ears.is_primary {
            // Forward is the transform's −Z axis (the coordinate contract).
            let (forward, up) = (-transform.matrix.col(2), transform.matrix.col(1));
            listener = ListenerState {
                position: transform.translation(),
                forward: Vec3::new(forward.x, forward.y, forward.z).normalize_or_zero(),
                up: Vec3::new(up.x, up.y, up.z).normalize_or_zero(),
                velocity: body.map_or(Vec3::ZERO, |body| body.linear_velocity),
            };
            listener_volume = ears.volume.max(0.0);
            break;
        }
    }
    let master = settings.master_volume.max(0.0) * listener_volume;
    if state.master != Some(master) && voices.control().set_bus_gain(MASTER, master).is_ok() {
        state.master = Some(master);
    }

    let reverb = reverb_bus(&mut state.reverb, &mut voices, &settings);
    let mode = settings.spatializer_mode;
    let traced = sync_worker(&mut state, &mut room, mode);
    let room_geometry = room
        .geometry
        .as_deref()
        .filter(|_| mode == SpatializerMode::Raytraced);
    let mut loudest = Loudest::default();
    let occlusion = if room_geometry.is_some() {
        None
    } else {
        occlusion.raycaster.as_deref()
    };
    let rate = f64::from(voices.sample_rate().max(1));
    let mut room_input = RoomInput {
        listener: listener.position,
        rays: settings.ray_count,
        max_bounces: settings.max_bounces,
        ..RoomInput::default()
    };
    for (entity, player, transform, body) in players.iter_mut() {
        let key = entity_key(entity);
        let emitter_reflections = traced
            .as_ref()
            .and_then(|traced| traced.reflections_of(key));
        let spatialized = transform.map(|transform| {
            let model: &mut dyn SpatialAudioModel = match mode {
                SpatializerMode::Standard => &mut state.standard,
                SpatializerMode::Raytraced => &mut state.raytraced,
            };
            spatialize(
                model,
                &listener,
                transform,
                body,
                &player.spatial,
                occlusion,
            )
        });
        let unoccluded_level =
            spatialized.map(|(spatial, _, _)| spatial.gain_left.max(spatial.gain_right));
        let spatial = spatialized.map(|(mut spatial, doppler, _)| {
            if let (Some(geometry), Some(transform)) = (room_geometry, transform) {
                apply_obstruction(
                    geometry,
                    emitter_reflections,
                    listener.position,
                    transform.translation(),
                    &mut spatial,
                );
            }
            (spatial, doppler)
        });
        let reverb = reverb.zip(spatialized.map(|(_, _, send)| send));
        let reflections = transform
            .zip(emitter_reflections)
            .map(|(transform, paths)| taps(paths, &listener, transform, &player.spatial));
        let mut cx = Context {
            voices: &mut voices,
            assets: &assets,
            clock,
            rate,
            tick_frames,
            spatial,
            reverb,
            reflections,
        };
        player.update(&mut cx);
        if let (Some(transform), Some(base_gain)) = (transform, unoccluded_level) {
            // The loudest sounding players next get reflections; silent ones only if there is
            // room, the nearest first.
            let sounding = if player.voices.is_empty() { 1e-3 } else { 1.0 };
            let level = player.volume * base_gain * sounding;
            loudest.offer(level, key, transform.translation());
        }
        if player.despawn_on_finish
            && player.played
            && player.voices.is_empty()
            && matches!(player.state, PlayerState::Finished | PlayerState::Stopped)
        {
            commands.despawn(entity);
        }
    }
    for (index, &(_, id, position)) in loudest.entries[..loudest.count].iter().enumerate() {
        room_input.receivers[index] = position;
        room_input.ids[index] = id;
    }
    room_input.receiver_count = loudest.count;
    if let Some(traced) = traced {
        follow_room(&mut state, &mut voices, &traced, room_input);
    }
}

/// Attenuates and filters `spatial` by `sqrt(transmission² + diffraction²)` across the room's
/// colliders when a surface blocks the direct line from `listener` to `emitter`.
fn apply_obstruction(
    geometry: &AcousticGeometry,
    reflections: Option<&EmitterReflections>,
    listener: Vec3,
    emitter: Vec3,
    spatial: &mut VoiceSpatial,
) {
    let delta = emitter - listener;
    let distance = delta.length();
    if distance <= 1e-4 {
        return;
    }
    let ray = AcousticRay::new(listener, delta / distance);
    if geometry.raycast_closest(&ray, distance - 1e-4).is_none() {
        return;
    }
    let transmission = geometry.raycast_transmission(&ray, distance);
    let diffraction = reflections.map_or([0.0; 3], |r| r.diffraction);
    let gain: [f32; 3] = std::array::from_fn(|band| {
        let t = transmission[band];
        let d = diffraction[band];
        (t * t + d * d).sqrt().min(1.0)
    });
    spatial.gain_left *= gain[0];
    spatial.gain_right *= gain[0];
    let cutoff = if gain[0] > 1e-6 {
        let r = (gain[2] / gain[0]).clamp(0.0, 1.0);
        if r >= 0.9999 {
            VoiceSpatial::OPEN
        } else if r <= 1e-4 {
            100.0
        } else {
            (4000.0 / ((1.0 - r * r) / (r * r)).powf(0.25)).clamp(100.0, VoiceSpatial::OPEN)
        }
    } else {
        100.0
    };
    spatial.cutoff_hz = spatial.cutoff_hz.min(cutoff);
}

/// The players with the highest levels this tick, up to [`MAX_RECEIVERS`]: the emitters the
/// acoustics worker traces next.
#[derive(Default)]
struct Loudest {
    entries: [(f32, u64, Vec3); MAX_RECEIVERS],
    count: usize,
}

impl Loudest {
    fn offer(&mut self, level: f32, id: u64, position: Vec3) {
        if self.count < MAX_RECEIVERS {
            self.entries[self.count] = (level, id, position);
            self.count += 1;
            return;
        }
        let quietest = (0..MAX_RECEIVERS)
            .min_by(|&a, &b| self.entries[a].0.total_cmp(&self.entries[b].0))
            .unwrap_or(0);
        if level > self.entries[quietest].0 {
            self.entries[quietest] = (level, id, position);
        }
    }
}

/// An entity's key for the acoustics worker: its index and generation.
fn entity_key(entity: Entity) -> u64 {
    (u64::from(entity.generation().get()) << 32) | u64::from(entity.index())
}

/// A player's reflection taps from its paths: each path's gain is the player's attenuation over
/// the path's length, its specular reflectance in the middle band and the player's cone in the
/// direction the path leaves it, panned by the direction it arrives from as the direct sound is.
fn taps(
    paths: &EmitterReflections,
    listener: &ListenerState,
    transform: &GlobalTransform,
    settings: &SpatialSettings,
) -> Reflections {
    let right = listener.forward.cross(listener.up).normalize_or_zero();
    let forward = -transform.matrix.col(2);
    let forward = Vec3::new(forward.x, forward.y, forward.z).normalize_or_zero();
    let panner = EqualPowerPanner::new();
    let mut reflections = Reflections::default();
    for path in &paths.paths[..paths.count] {
        let mut gain =
            settings.attenuation.calculate_gain(path.length) * path.reflectance[1].max(0.0).sqrt();
        if let Some(cone) = &settings.cone {
            let off_axis = forward.dot(path.departure).clamp(-1.0, 1.0).acos();
            gain *= cone.calculate_attenuation(off_axis).0;
        }
        let azimuth = path
            .arrival
            .dot(right)
            .atan2(path.arrival.dot(listener.forward));
        let (left, right) = panner.calculate_gains(azimuth);
        reflections.taps[reflections.count] = ReflectionTap {
            delay: path.delay,
            gain_left: gain * left,
            gain_right: gain * right,
        };
        reflections.count += 1;
    }
    reflections
}

/// Keeps raytraced mode's acoustics worker on the room's geometry and returns its latest
/// output, with the room's decay time and traces published. Outside raytraced mode, or without
/// a geometry, the worker stops.
fn sync_worker(
    state: &mut PlayerSystemState,
    room: &mut AudioRoom,
    mode: SpatializerMode,
) -> Option<RoomOutput> {
    let Some(geometry) = room
        .geometry
        .as_ref()
        .filter(|_| mode == SpatializerMode::Raytraced)
    else {
        state.worker = None;
        room.rt60 = None;
        room.traces = 0;
        return None;
    };
    if state
        .worker
        .as_ref()
        .is_none_or(|worker| !Arc::ptr_eq(worker.geometry(), geometry))
    {
        state.worker = Some(RoomWorker::spawn(Arc::clone(geometry)));
    }
    let output = state.worker.as_mut()?.output();
    room.rt60 = output.rt60;
    room.traces = output.traces;
    Some(output)
}

/// Hands the acoustics worker this tick's listener and emitters, and gives the late reverb the
/// decay time it measured (when it moves by more than 1 %).
fn follow_room(
    state: &mut PlayerSystemState,
    voices: &mut AudioVoices,
    traced: &RoomOutput,
    input: RoomInput,
) {
    if let Some(worker) = state.worker.as_mut() {
        worker.update(input);
    }
    let (Some(rt60), Some(reverb)) = (traced.rt60, state.reverb.as_mut()) else {
        return;
    };
    let moved = reverb.rt60.is_none_or(|sent| {
        sent.iter()
            .zip(&rt60)
            .any(|(sent, now)| (now / sent - 1.0).abs() > 0.01)
    });
    if moved {
        sent(voices.control().set_reverb_time(reverb.bus, rt60));
        reverb.rt60 = Some(rt60);
    }
}

/// A player's stereo gains, cutoff, Doppler factor and reverb send from its transform and the
/// listener.
fn spatialize(
    model: &mut dyn SpatialAudioModel,
    listener: &ListenerState,
    transform: &GlobalTransform,
    body: Option<&RigidBody>,
    settings: &SpatialSettings,
    occlusion: Option<&dyn AcousticRaycaster>,
) -> (VoiceSpatial, f32, f32) {
    // Forward, where a cone points, is the transform's −Z axis (the coordinate contract).
    let forward = -transform.matrix.col(2);
    let emitter = EmitterState {
        position: transform.translation(),
        forward: Vec3::new(forward.x, forward.y, forward.z).normalize_or_zero(),
        velocity: body.map_or(Vec3::ZERO, |body| body.linear_velocity),
        attenuation: settings.attenuation,
        cone: settings.cone,
        doppler_factor: settings.doppler_factor,
    };
    let mut params = VoiceSpatialParams::default();
    model.spatialize_emitter(listener, &emitter, occlusion, &mut params);
    let spatial = VoiceSpatial {
        gain_left: params.gain_left,
        gain_right: params.gain_right,
        cutoff_hz: params.low_pass_cutoff,
        delay_left: params.delay_left_sec,
        delay_right: params.delay_right_sec,
        shadow_left_hz: params.head_shadow_left,
        shadow_right_hz: params.head_shadow_right,
    };
    (spatial, params.pitch_scale, params.reverb_send)
}

impl AudioPlayer {
    /// One tick: drops the voices the mixer ended, follows the player's position in space,
    /// applies its requests, schedules its playlist and refreshes its state.
    fn update(&mut self, cx: &mut Context<'_>) {
        let voices = &*cx.voices;
        self.voices.retain(|voice| voices.is_playing(voice.handle));
        if self.autoplay && !self.played && !self.requests.contains(&Some(Request::Play)) {
            self.request(Request::Play);
        }
        self.follow(cx);
        self.reflect(cx);
        self.apply_requests(cx);
        self.schedule_next(cx);
        self.refresh(cx);
    }

    /// Sends the voices their stereo gains, cutoff and Doppler pitch, ramped over a tick.
    fn follow(&mut self, cx: &mut Context<'_>) {
        let Some((spatial, doppler)) = cx.spatial else {
            return;
        };
        let doppler_changed = (doppler - self.doppler).abs() > 1e-4;
        self.doppler = doppler;
        for voice in &mut self.voices {
            sent(cx.voices.set_spatial(voice.handle, spatial, cx.tick_frames));
            if let Some((bus, level)) = cx.reverb {
                sent(cx.voices.set_send(voice.handle, bus, level, cx.tick_frames));
            }
            if doppler_changed {
                sent(
                    cx.voices
                        .set_pitch(voice.handle, self.pitch * doppler, cx.tick_frames),
                );
                voice.pitch_changed = true;
            }
        }
    }

    /// Gives the voices their early reflections, ramped over a tick, or takes them away once
    /// the player has none.
    fn reflect(&mut self, cx: &mut Context<'_>) {
        let (reflections, frames) = match cx.reflections {
            Some(reflections) => (reflections, cx.tick_frames),
            None if self.reflecting => (Reflections::default(), 0),
            None => return,
        };
        for voice in &self.voices {
            sent(cx.voices.set_reflections(voice.handle, reflections, frames));
        }
        self.reflecting = cx.reflections.is_some();
    }

    /// Applies the requests in call order. A play request whose clips are still loading waits,
    /// and the requests after it with it.
    fn apply_requests(&mut self, cx: &mut Context<'_>) {
        let mut kept = 0;
        for index in 0..self.requests.len() {
            let Some(request) = self.requests[index].take() else {
                break;
            };
            if kept > 0 || !self.apply(request, cx) {
                self.requests[kept] = Some(request);
                kept += 1;
            }
        }
    }

    /// Applies one request; false if it must wait for its clips to load.
    fn apply(&mut self, request: Request, cx: &mut Context<'_>) -> bool {
        match request {
            Request::Play => return self.start(cx),
            Request::Stop { fade } => {
                let fade = Fade::linear(cx.frames(fade));
                for voice in &mut self.voices {
                    sent(cx.voices.stop_with(voice.handle, At::Now, fade));
                    voice.succeeded = true;
                }
                self.stopping = true;
                self.paused = false;
            }
            Request::Pause => {
                for voice in &self.voices {
                    sent(cx.voices.pause(voice.handle));
                }
                self.paused = true;
            }
            Request::Resume => {
                for voice in &self.voices {
                    sent(cx.voices.resume(voice.handle));
                }
                self.paused = false;
            }
            Request::Transition { item, crossfade } => return self.transition(item, crossfade, cx),
            Request::Volume { volume, ramp } => {
                self.volume = volume;
                let frames = cx.frames(ramp);
                for index in 0..self.voices.len() {
                    let voice = self.voices[index];
                    sent(
                        cx.voices
                            .set_gain(voice.handle, self.gain_of(voice.item), frames),
                    );
                }
            }
            Request::Pitch { pitch, ramp } => {
                self.pitch = pitch;
                let frames = cx.frames(ramp);
                for voice in &mut self.voices {
                    sent(
                        cx.voices
                            .set_pitch(voice.handle, pitch * self.doppler, frames),
                    );
                    voice.pitch_changed = true;
                }
            }
            Request::StemVolume { stem, volume, ramp } => {
                if let Some(level) = self.stem_volumes.get_mut(stem) {
                    *level = volume;
                }
                let frames = cx.frames(ramp);
                for index in 0..self.voices.len() {
                    let voice = self.voices[index];
                    if voice.item == stem {
                        sent(cx.voices.set_gain(voice.handle, self.gain_of(stem), frames));
                    }
                }
            }
        }
        true
    }

    /// The gain of a voice playing `item`: the volume, times the stem's for stems.
    fn gain_of(&self, item: usize) -> f32 {
        match self.source {
            AudioSource::Stems { .. } => self.volume * self.stem_volumes[item],
            _ => self.volume,
        }
    }

    /// The parameters of a new voice playing `item`.
    fn params(&self, item: usize, cx: &mut Context<'_>) -> VoiceParams {
        let (looping, loop_start, loop_end) = match self.source {
            AudioSource::Clip {
                looping,
                loop_start,
                loop_end,
                ..
            } => (looping, loop_start, loop_end),
            AudioSource::Stems { looping, .. } => (looping, 0, None),
            AudioSource::Playlist { .. } => (false, 0, None),
        };
        VoiceParams {
            gain: self.gain_of(item),
            pitch: self.pitch * self.doppler,
            looping,
            bus: cx.voices.control().bus(&self.bus).unwrap_or(MASTER),
            priority: self.priority,
            loop_start,
            loop_end,
        }
    }

    /// Starts a voice of `clip` playing `item` at `at`, fading in over `fade_in`.
    fn start_voice(
        &mut self,
        clip: RegisteredClip,
        item: usize,
        at: At,
        fade_in: Fade,
        cx: &mut Context<'_>,
    ) -> Option<VoiceHandle> {
        let params = self.params(item, cx);
        match cx.voices.start_with(clip.id, params, at, fade_in) {
            Ok(handle) => {
                if let Some((spatial, _)) = cx.spatial {
                    // Before it sounds, so it starts in place.
                    sent(cx.voices.set_spatial(handle, spatial, 0));
                }
                if let Some((bus, level)) = cx.reverb {
                    sent(cx.voices.set_send(handle, bus, level, 0));
                }
                if let Some(reflections) = cx.reflections {
                    sent(cx.voices.set_reflections(handle, reflections, 0));
                }
                self.voices.push(PlayerVoice {
                    handle,
                    item,
                    start_pitch: params.pitch,
                    pitch_changed: false,
                    succeeded: false,
                });
                Some(handle)
            }
            Err(error) => {
                tracing::warn!("an audio player could not start a voice: {error}");
                None
            }
        }
    }

    /// Stops every voice still playing an item, over `fade`.
    fn stop_all(&mut self, fade: Fade, cx: &mut Context<'_>) {
        for voice in &mut self.voices {
            if !voice.succeeded {
                sent(cx.voices.stop_with(voice.handle, At::Now, fade));
                voice.succeeded = true;
            }
        }
    }

    /// Plays the source from its start; false while its clips are loading.
    fn start(&mut self, cx: &mut Context<'_>) -> bool {
        match &self.source {
            AudioSource::Clip { clip, .. } => {
                let Some(clip) = cx.clip(clip) else {
                    return false;
                };
                while self.voices.len() >= self.max_instances {
                    let oldest = self.voices.remove(0);
                    sent(cx.voices.stop(oldest.handle));
                }
                self.start_voice(clip, 0, At::Now, Fade::NONE, cx);
            }
            AudioSource::Playlist { items, shuffle, .. } => {
                let (count, shuffle) = (items.len(), *shuffle);
                if count == 0 {
                    return true;
                }
                let first = if shuffle { self.random(count) } else { 0 };
                let AudioSource::Playlist { items, .. } = &self.source else {
                    unreachable!()
                };
                let Some(clip) = cx.clip(&items[first]) else {
                    return false;
                };
                self.stop_all(Fade::NONE, cx);
                self.current_item = first;
                self.start_voice(clip, first, At::Now, Fade::NONE, cx);
            }
            AudioSource::Stems { stems, .. } => {
                // Every stem must be ready, so that they start on the same frame.
                let mut ready = true;
                for stem in stems {
                    ready &= cx.clip(stem).is_some();
                }
                if !ready {
                    return false;
                }
                let count = stems.len();
                self.stop_all(Fade::NONE, cx);
                for stem in 0..count {
                    let AudioSource::Stems { stems, .. } = &self.source else {
                        unreachable!()
                    };
                    if let Some(clip) = cx.clip(&stems[stem]) {
                        self.start_voice(clip, stem, At::Now, Fade::NONE, cx);
                    }
                }
            }
        }
        self.played = true;
        self.stopping = false;
        self.paused = false;
        self.state = PlayerState::Playing;
        true
    }

    /// Moves a playlist to `item` with an equal-power crossfade; false while it is loading.
    fn transition(&mut self, item: usize, crossfade: f32, cx: &mut Context<'_>) -> bool {
        let AudioSource::Playlist { items, .. } = &self.source else {
            return true;
        };
        let Some(handle) = items.get(item) else {
            return true;
        };
        let Some(clip) = cx.clip(handle) else {
            return false;
        };
        let frames = cx.frames(crossfade);
        let fade = if frames == 0 {
            Fade::NONE
        } else {
            Fade::equal_power(frames)
        };
        self.stop_all(fade, cx);
        self.current_item = item;
        self.start_voice(clip, item, At::Now, fade, cx);
        self.played = true;
        self.stopping = false;
        self.paused = false;
        self.state = PlayerState::Playing;
        true
    }

    /// Schedules a playlist's next item on the frame the current one has its crossfade left,
    /// once that frame is less than [`SCHEDULE_AHEAD`] away.
    fn schedule_next(&mut self, cx: &mut Context<'_>) {
        if self.paused || self.stopping {
            return;
        }
        let AudioSource::Playlist {
            items,
            crossfade,
            looping,
            shuffle,
        } = &self.source
        else {
            return;
        };
        let Some(index) = self.voices.iter().rposition(|voice| !voice.succeeded) else {
            return;
        };
        let voice = self.voices[index];
        let Some(current) = cx.clip(&items[voice.item]) else {
            return;
        };
        // Output frames the item lasts at the pitch it started with.
        let rate = f64::from(current.sample_rate.max(1));
        let pitch = f64::from(voice.start_pitch.max(1e-3));
        let length = current.frames as f64 * cx.rate / (rate * pitch);
        let end = if voice.pitch_changed {
            // A changed pitch: from the position, to within a device period.
            let Some(position) = cx.voices.position(voice.handle) else {
                return;
            };
            let pitch = f64::from((self.pitch * self.doppler).max(1e-3));
            let left = (current.frames as f64 / rate - position).max(0.0);
            cx.clock as f64 + left * cx.rate / pitch
        } else {
            let Some(started) = cx.voices.started_at(voice.handle) else {
                return;
            };
            started as f64 + length
        };
        let fade = (f64::from(*crossfade) * cx.rate).min(length).round();
        let at = (end - fade).round().max(cx.clock as f64);
        if at > cx.clock as f64 + SCHEDULE_AHEAD * cx.rate {
            return;
        }
        let count = items.len();
        let (looping, shuffle) = (*looping, *shuffle);
        let next = if shuffle && count > 1 {
            let mut next = self.random(count - 1);
            if next >= voice.item {
                next += 1;
            }
            Some(next)
        } else if voice.item + 1 < count {
            Some(voice.item + 1)
        } else if looping {
            Some(0)
        } else {
            None
        };
        let Some(next) = next else {
            // The last item ends by itself.
            self.voices[index].succeeded = true;
            return;
        };
        let AudioSource::Playlist { items, .. } = &self.source else {
            return;
        };
        let Some(clip) = cx.clip(&items[next]) else {
            return;
        };
        let at = At::Frame(at as u64);
        let fade = fade as u32;
        let fade = if fade == 0 {
            Fade::NONE
        } else {
            sent(
                cx.voices
                    .stop_with(voice.handle, at, Fade::equal_power(fade)),
            );
            Fade::equal_power(fade)
        };
        self.voices[index].succeeded = true;
        self.start_voice(clip, next, at, fade, cx);
    }

    /// The state, position and current item from the mixer.
    fn refresh(&mut self, cx: &mut Context<'_>) {
        if self.voices.is_empty() {
            if matches!(self.state, PlayerState::Playing | PlayerState::Paused) {
                self.state = if self.stopping {
                    PlayerState::Stopped
                } else {
                    PlayerState::Finished
                };
            }
        } else {
            self.state = if self.paused {
                PlayerState::Paused
            } else {
                PlayerState::Playing
            };
        }
        for voice in self.voices.iter().rev() {
            if cx.voices.started_at(voice.handle).is_some() {
                self.current_item = voice.item;
                self.position = cx.voices.position(voice.handle).unwrap_or(0.0);
                break;
            }
        }
    }

    /// A pseudo-random index below `count` (xorshift), for shuffled playlists.
    fn random(&mut self, count: usize) -> usize {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng % count as u64) as usize
    }
}
