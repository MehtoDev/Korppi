//! High-performance, modular 3D game audio engine supporting both standard
//! psychoacoustic spatial audio and physically simulated raytraced acoustics.

#![warn(missing_docs)]

pub mod asset;
pub mod backend;
pub mod coordinator;
pub mod device;
pub mod dsp;
pub mod ecs;
pub mod error;
pub mod frame;
pub mod mixer;
pub mod output;
pub mod player;
pub mod raytrace;
pub mod ring_buffer;
pub mod spatial;
pub mod synthesis;

pub use asset::{AudioClip, AudioClipLoader, OggVorbisLoader, WavAudioLoader};
pub use coordinator::AudioCoordinator;
pub use device::{
    AudioCallback, AudioCallbackInfo, AudioDevice, AudioStream, AudioStreamConfig, BufferSize,
    ChannelCount, HeadlessDummyDevice,
};
pub use dsp::{BiquadFilter, BiquadType, Freeverb, PeakLimiter};
pub use ecs::{AcousticMaterialComponent, AudioListener, AudioSettings, RoomReverbSettings};
pub use error::{AudioError, AudioResult};
pub use frame::AudioFrame;
pub use player::{AudioOcclusion, AudioPlayer, AudioRoom, AudioSource, PlayerState};
pub use raytrace::{
    eyring_rt60, fibonacci_sphere_rays, sabine_rt60, AcousticBox, AcousticHit, AcousticMaterial,
    AcousticRay, AcousticScene, AcousticTriangle, EarlyReflectionHit, EnergyHistogram,
    PathTraceConfig, RaytracedAcoustics,
};
pub use ring_buffer::{Consumer, Producer, SpscRingBuffer};
pub use spatial::{
    AcousticRaycaster, AttenuationModel, DistanceAttenuation, EmitterCone, EmitterState,
    EqualPowerPanner, ListenerState, SpatialAudioModel, SpatializerMode, StandardSpatializer,
    VoiceSpatialParams, WoodworthBinaural,
};
pub use synthesis::{
    householder_diffuse_8, triple_buffer, triple_buffer_with, EarlyReflectionTap, FdnConfig,
    FeedbackDelayNetwork, TappedDelayLine, TripleBufferConsumer, TripleBufferProducer,
};
