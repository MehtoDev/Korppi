//! A game for tests of the player path: the ECS world, the asset server, the mixer and a headless
//! stream, ticked like the engine ticks them (transforms, then the audio players, then a tick of
//! output).

#![allow(dead_code)]

use korppi_asset::{AssetHandle, AssetServer, MemoryFileSystem, Vfs};
use korppi_audio::mixer::{AudioMixer, AudioVoices, MixerConfig};
use korppi_audio::player::{update_audio_players, AudioOcclusion, AudioPlayer, AudioRoom};
use korppi_audio::{
    AudioClip, AudioClipLoader, AudioDevice, AudioSettings, AudioStream, AudioStreamConfig,
    BufferSize, ChannelCount, HeadlessDummyDevice,
};
use korppi_ecs::{Entity, Schedule, Stage, World};
use korppi_scene::propagate_transforms;

pub const RATE: u32 = 48_000;
/// Output frames per game tick (60 Hz).
pub const TICK: usize = 800;

pub struct Game {
    pub world: World,
    schedule: Schedule,
    pub stream: Box<dyn AudioStream>,
    files: Vec<(String, Vec<u8>)>,
}

impl Game {
    pub fn new() -> Self {
        let config = AudioStreamConfig {
            sample_rate: RATE,
            channels: ChannelCount::Stereo,
            buffer_size: BufferSize::Fixed(256),
        };
        let (control, mixer) = AudioMixer::new(MixerConfig::for_stream(&config));
        let mut stream = HeadlessDummyDevice::new()
            .create_output_stream(config, Box::new(mixer))
            .expect("stream");
        stream.play().expect("play");
        let mut world = World::new();
        world.insert_resource(AudioVoices::new(control));
        world.insert_resource(AudioSettings::default());
        world.insert_resource(AudioOcclusion::default());
        world.insert_resource(AudioRoom::default());
        let mut schedule = Schedule::new();
        schedule.add(Stage::PostUpdate, update_audio_players);
        Self {
            world,
            schedule,
            stream,
            files: Vec::new(),
        }
    }

    /// Adds `clip` as the WAV file `name`; [`Self::start`] loads every file through the asset
    /// server.
    pub fn clip(&mut self, name: &str, clip: AudioClip) -> String {
        self.file(&format!("{name}.wav"), clip.to_wav_bytes())
    }

    /// Adds the file `path` with `bytes` (WAV or Ogg Vorbis).
    pub fn file(&mut self, path: &str, bytes: Vec<u8>) -> String {
        self.files.push((path.to_string(), bytes));
        path.to_string()
    }

    /// Mounts the clips and returns their handles, in the order added.
    pub fn start(&mut self) -> Vec<AssetHandle<AudioClip>> {
        let mut files = MemoryFileSystem::new();
        for (path, bytes) in &self.files {
            files.insert(path, bytes.clone());
        }
        let mut vfs = Vfs::new();
        vfs.mount("", Box::new(files));
        let assets = AssetServer::new(vfs);
        assets.register_loader(AudioClipLoader);
        let handles = self
            .files
            .iter()
            .map(|(path, _)| assets.load_sync::<AudioClipLoader>(path).expect("load"))
            .collect();
        self.world.insert_resource(assets);
        handles
    }

    /// One game tick: the players update, then the device renders a tick of output.
    pub fn tick(&mut self) -> Vec<f32> {
        propagate_transforms(&mut self.world);
        self.schedule.run(&mut self.world);
        self.stream.render_frames(TICK)
    }

    pub fn ticks(&mut self, count: usize) -> Vec<f32> {
        (0..count).flat_map(|_| self.tick()).collect()
    }

    pub fn player(&mut self, entity: Entity) -> &mut AudioPlayer {
        self.world.get_mut::<AudioPlayer>(entity).expect("player")
    }
}

pub fn constant(value: f32, frames: usize) -> AudioClip {
    AudioClip::new(1, RATE, vec![value; frames])
}

/// One channel of interleaved stereo.
pub fn channel(out: &[f32], channel: usize) -> Vec<f32> {
    out.iter().skip(channel).step_by(2).copied().collect()
}

/// The frequency of a tone, from the rising zero crossings between the first and the last.
pub fn frequency(samples: &[f32]) -> f32 {
    let rising: Vec<usize> = samples
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] < 0.0 && w[1] >= 0.0)
        .map(|(i, _)| i)
        .collect();
    let (first, last) = (rising[0], rising[rising.len() - 1]);
    (rising.len() - 1) as f32 * RATE as f32 / (last - first) as f32
}

/// The RMS level in dB (full scale 0 dB).
pub fn rms_db(samples: &[f32]) -> f32 {
    let mean = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    10.0 * mean.max(1e-20).log10()
}
