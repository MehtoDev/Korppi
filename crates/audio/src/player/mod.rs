//! The gameplay audio API: [`AudioPlayer`] components (FIX-7.2, ADR-050; finding AUD-07).
//!
//! A player holds a source ([`AudioSource`]: a clip, optionally with an intro and a loop region;
//! a playlist with crossfades; or layered stems), its playback settings, and the requests
//! gameplay code makes through its controls ([`AudioPlayer::play`], [`AudioPlayer::stop`],
//! [`AudioPlayer::pause`], ...). [`update_audio_players`], run once per game tick, turns the
//! requests into mixer voices through the [`AudioVoices`](crate::mixer::AudioVoices) resource,
//! schedules playlist transitions on the audio clock to the sample, spatializes players whose
//! entity has a `GlobalTransform` against the primary [`AudioListener`](crate::AudioListener),
//! and writes the mixer's state back into each player ([`AudioPlayer::state`],
//! [`AudioPlayer::position`]).
//!
//! An entity holds one player; several sounds on one object are players on child entities,
//! which the hierarchy positions with their parent.
//!
//! Inputs: the players, the listener, `AudioVoices`, the `AssetServer` (clips are named by asset
//! handle and registered with the mixer the first time they play), `AudioSettings` (master
//! volume and spatializer mode) and [`AudioOcclusion`] (what blocks the line of sight). Outputs: mixer commands, each player's state, and the
//! despawning of finished players that ask for it.

mod system;

pub use system::update_audio_players;

use crate::asset::AudioClip;
use crate::mixer::VoiceHandle;
use crate::spatial::{AcousticRaycaster, AttenuationModel, EmitterCone};
use korppi_asset::AssetHandle;
use korppi_ecs::Component;
use std::sync::Arc;

/// What blocks the line of sight between spatialized players and the listener: a world resource
/// read by [`update_audio_players`]. An occluded player is attenuated and muffled.
#[derive(Clone, Default)]
pub struct AudioOcclusion {
    /// The geometry rays are cast against (an `AcousticScene`, or the game's own), or none.
    pub raycaster: Option<Arc<dyn AcousticRaycaster>>,
}

/// Raytraced mode's room: the geometry the acoustics worker traces (the engine sets it to the
/// level's static colliders) and the decay time the worker measured. A world resource read and
/// written by [`update_audio_players`].
#[derive(Default)]
pub struct AudioRoom {
    /// The level's acoustic geometry, if any.
    pub geometry: Option<Arc<crate::raytrace::AcousticGeometry>>,
    rt60: Option<[f32; 3]>,
    traces: u64,
}

impl AudioRoom {
    /// The room's decay time per band (seconds), once the worker has measured it in raytraced
    /// mode; the late reverb decays at it.
    #[must_use]
    pub fn rt60(&self) -> Option<[f32; 3]> {
        self.rt60
    }

    /// The traces the worker has completed of the current geometry (0 outside raytraced mode).
    #[must_use]
    pub fn traces(&self) -> u64 {
        self.traces
    }
}

/// Requests a player holds between two updates.
const MAX_REQUESTS: usize = 16;

/// What a player plays.
#[derive(Clone, Debug)]
pub enum AudioSource {
    /// One clip.
    Clip {
        /// The clip.
        clip: AssetHandle<AudioClip>,
        /// Whether it repeats.
        looping: bool,
        /// Where a loop returns to, in clip frames: the frames before it are an intro.
        loop_start: u32,
        /// Where a loop returns from, in clip frames; `None` is the clip's end.
        loop_end: Option<u32>,
    },
    /// Clips played one after another.
    Playlist {
        /// The clips, in order.
        items: Vec<AssetHandle<AudioClip>>,
        /// The equal-power crossfade between items, in seconds; 0 plays them gaplessly.
        crossfade: f32,
        /// Whether the list starts over after its last item.
        looping: bool,
        /// Whether items play in a shuffled order instead of in order (never the same item
        /// twice in a row).
        shuffle: bool,
    },
    /// Layered clips that start on the same frame and stay in sync, each with its own volume.
    Stems {
        /// The layers.
        stems: Vec<AssetHandle<AudioClip>>,
        /// Whether they repeat.
        looping: bool,
    },
}

impl AudioSource {
    /// `clip`, played once.
    #[must_use]
    pub fn clip(clip: AssetHandle<AudioClip>) -> Self {
        Self::Clip {
            clip,
            looping: false,
            loop_start: 0,
            loop_end: None,
        }
    }

    /// `items` in order, crossfading over `crossfade` seconds, played once.
    #[must_use]
    pub fn playlist(items: Vec<AssetHandle<AudioClip>>, crossfade: f32) -> Self {
        Self::Playlist {
            items,
            crossfade: crossfade.max(0.0),
            looping: false,
            shuffle: false,
        }
    }

    /// `stems` together, played once.
    #[must_use]
    pub fn stems(stems: Vec<AssetHandle<AudioClip>>) -> Self {
        Self::Stems {
            stems,
            looping: false,
        }
    }

    /// The source, repeating.
    #[must_use]
    pub fn looping(mut self) -> Self {
        match &mut self {
            Self::Clip { looping, .. }
            | Self::Playlist { looping, .. }
            | Self::Stems { looping, .. } => *looping = true,
        }
        self
    }

    /// A clip source looping from `start` to `end` (clip frames, `None` for the clip's end)
    /// after playing the frames before `start` once. Other sources are unchanged.
    #[must_use]
    pub fn loop_region(mut self, start: u32, end: Option<u32>) -> Self {
        if let Self::Clip {
            loop_start,
            loop_end,
            ..
        } = &mut self
        {
            *loop_start = start;
            *loop_end = end;
        }
        self
    }

    /// A playlist in shuffled order. Other sources are unchanged.
    #[must_use]
    pub fn shuffled(mut self) -> Self {
        if let Self::Playlist { shuffle, .. } = &mut self {
            *shuffle = true;
        }
        self
    }

    /// How many voices the source plays at once at most, besides crossfades.
    fn layers(&self) -> usize {
        match self {
            Self::Stems { stems, .. } => stems.len(),
            _ => 1,
        }
    }
}

/// How a player sounds in space, for players on entities with a `GlobalTransform`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpatialSettings {
    /// Distance attenuation.
    pub attenuation: AttenuationModel,
    /// A directional cone, or none for a source that radiates equally.
    pub cone: Option<EmitterCone>,
    /// Doppler strength: 0 off, 1 physical.
    pub doppler_factor: f32,
}

impl Default for SpatialSettings {
    fn default() -> Self {
        Self {
            attenuation: AttenuationModel::default(),
            cone: None,
            doppler_factor: 1.0,
        }
    }
}

/// Where a player is in its life.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PlayerState {
    /// Not played yet, or stopped by [`AudioPlayer::stop`].
    Stopped,
    /// Playing.
    Playing,
    /// Paused by [`AudioPlayer::pause`].
    Paused,
    /// The source played to its end.
    Finished,
}

/// A control call, applied at the next update in call order.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Request {
    Play,
    Stop { fade: f32 },
    Pause,
    Resume,
    Transition { item: usize, crossfade: f32 },
    Volume { volume: f32, ramp: f32 },
    Pitch { pitch: f32, ramp: f32 },
    StemVolume { stem: usize, volume: f32, ramp: f32 },
}

/// A voice a player plays, and what scheduling knows of it.
#[derive(Copy, Clone, Debug)]
struct PlayerVoice {
    handle: VoiceHandle,
    /// The playlist item or stem it plays.
    item: usize,
    /// The pitch it started with; scheduling from its start frame holds while it keeps it.
    start_pitch: f32,
    pitch_changed: bool,
    /// Whether the item after it has been scheduled, or it is fading out.
    succeeded: bool,
}

/// Plays sound for its entity: the default way for gameplay code to play audio (see the module
/// documentation).
#[derive(Clone, Debug)]
pub struct AudioPlayer {
    source: AudioSource,
    volume: f32,
    pitch: f32,
    bus: String,
    priority: u8,
    max_instances: usize,
    autoplay: bool,
    despawn_on_finish: bool,
    /// How the player sounds in space.
    pub spatial: SpatialSettings,
    requests: [Option<Request>; MAX_REQUESTS],
    // The state the update keeps.
    state: PlayerState,
    voices: Vec<PlayerVoice>,
    stem_volumes: Vec<f32>,
    current_item: usize,
    position: f64,
    played: bool,
    stopping: bool,
    paused: bool,
    /// The Doppler pitch factor last applied.
    doppler: f32,
    /// Whether its voices were last given early reflections.
    reflecting: bool,
    rng: u64,
}

impl Component for AudioPlayer {}

impl AudioPlayer {
    /// A player of `source` at full volume on the "Master" bus, waiting for [`Self::play`].
    #[must_use]
    pub fn new(source: AudioSource) -> Self {
        let layers = source.layers();
        Self {
            source,
            volume: 1.0,
            pitch: 1.0,
            bus: "Master".to_string(),
            priority: crate::mixer::DEFAULT_PRIORITY,
            max_instances: 1,
            autoplay: false,
            despawn_on_finish: false,
            spatial: SpatialSettings::default(),
            requests: [None; MAX_REQUESTS],
            state: PlayerState::Stopped,
            // Room for every layer and a crossfade, so that playing never grows it.
            voices: Vec::with_capacity(layers + 2),
            stem_volumes: vec![1.0; layers],
            current_item: 0,
            position: 0.0,
            played: false,
            stopping: false,
            paused: false,
            doppler: 1.0,
            reflecting: false,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Plays as soon as the player's clips are loaded, without a [`Self::play`] call.
    #[must_use]
    pub fn autoplay(mut self) -> Self {
        self.autoplay = true;
        self
    }

    /// Despawns the player's entity once it has played and stopped or finished: for one-shots.
    #[must_use]
    pub fn despawn_on_finish(mut self) -> Self {
        self.despawn_on_finish = true;
        self
    }

    /// The volume to start at (linear).
    #[must_use]
    pub fn with_volume(mut self, volume: f32) -> Self {
        self.volume = volume.max(0.0);
        self
    }

    /// The pitch to start at (2 is an octave up).
    #[must_use]
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.pitch = pitch.max(0.0);
        self
    }

    /// The mixer bus to play into, by name; an unknown name plays into "Master".
    #[must_use]
    pub fn with_bus(mut self, bus: &str) -> Self {
        self.bus = bus.to_string();
        self
    }

    /// The voices' priority: higher is mixed first, and replaced last when the pool is full.
    #[must_use]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// How many instances of a clip source may overlap: [`Self::play`] beyond it replaces the
    /// oldest. 1 by default.
    #[must_use]
    pub fn max_instances(mut self, instances: usize) -> Self {
        self.max_instances = instances.max(1);
        // Room for every instance and a crossfade, so that playing never grows it.
        self.voices.reserve(self.max_instances + 2);
        self
    }

    /// The source.
    #[must_use]
    pub fn source(&self) -> &AudioSource {
        &self.source
    }

    /// Plays the source from its start. A clip source adds an instance, up to
    /// [`Self::max_instances`]; a playlist starts at its first item; stems start together.
    pub fn play(&mut self) {
        self.request(Request::Play);
    }

    /// Stops, fading out over `fade` seconds (at least 5 ms).
    pub fn stop(&mut self, fade: f32) {
        self.request(Request::Stop {
            fade: fade.max(0.0),
        });
    }

    /// Pauses, fading out over 5 ms; [`Self::resume`] continues from the same place.
    pub fn pause(&mut self) {
        self.request(Request::Pause);
    }

    /// Resumes a paused player, fading in over 5 ms.
    pub fn resume(&mut self) {
        self.request(Request::Resume);
    }

    /// Moves a playlist to `item`, crossfading over `crossfade` seconds.
    pub fn transition_to(&mut self, item: usize, crossfade: f32) {
        self.request(Request::Transition {
            item,
            crossfade: crossfade.max(0.0),
        });
    }

    /// Ramps the volume to `volume` (linear) over `ramp` seconds (at least 5 ms).
    pub fn set_volume(&mut self, volume: f32, ramp: f32) {
        self.request(Request::Volume {
            volume: volume.max(0.0),
            ramp: ramp.max(0.0),
        });
    }

    /// Ramps the pitch to `pitch` over `ramp` seconds (at least 5 ms).
    pub fn set_pitch(&mut self, pitch: f32, ramp: f32) {
        self.request(Request::Pitch {
            pitch: pitch.max(0.0),
            ramp: ramp.max(0.0),
        });
    }

    /// Ramps stem `stem`'s volume to `volume` (linear) over `ramp` seconds (at least 5 ms).
    pub fn set_stem_volume(&mut self, stem: usize, volume: f32, ramp: f32) {
        self.request(Request::StemVolume {
            stem,
            volume: volume.max(0.0),
            ramp: ramp.max(0.0),
        });
    }

    /// The player's state, as of the last update.
    #[must_use]
    pub fn state(&self) -> PlayerState {
        self.state
    }

    /// Whether the player is playing (not paused, stopped or finished), as of the last update.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.state == PlayerState::Playing
    }

    /// The position in the current clip, in seconds, as of the last update.
    #[must_use]
    pub fn position(&self) -> f64 {
        self.position
    }

    /// The playlist item playing (0 for other sources).
    #[must_use]
    pub fn current_item(&self) -> usize {
        self.current_item
    }

    /// The volume, as last set.
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.volume
    }

    fn request(&mut self, request: Request) {
        if let Some(slot) = self.requests.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(request);
        } else {
            tracing::warn!(
                "an AudioPlayer holds {MAX_REQUESTS} requests already; dropped {request:?}"
            );
        }
    }
}
