//! Raytraced geometric acoustics, Monte Carlo path tracing, diffraction, and energy decay simulation.

pub mod acoustic_geometry;
pub mod diffraction;
pub mod energy_decay;
pub mod geometry;
pub mod image_source;
pub mod material;
pub mod path_tracer;
pub mod room_worker;
pub mod tracer;

pub use acoustic_geometry::AcousticGeometry;
pub use diffraction::{diffracted_sound, EdgeResponse, Wedge, DIFFRACTION_BANDS};
pub use energy_decay::{eyring_rt60, sabine_rt60, EnergyHistogram};
pub use geometry::{AcousticBox, AcousticHit, AcousticRay, AcousticScene, AcousticTriangle};
pub use image_source::{ImagePath, ReflectingPlane, SpecularChain, MAX_IMAGE_ORDER};
pub use material::AcousticMaterial;
pub use path_tracer::{PathTracer, ReflectionPath, RoomResponse, TraceSettings, TraceStatistics};
pub use room_worker::{
    EarlyPath, EmitterReflections, RoomInput, RoomOutput, RoomWorker, MAX_RECEIVERS,
};
pub use tracer::{
    fibonacci_sphere_rays, EarlyReflectionHit, PathTraceConfig, RaytracedAcoustics,
    AIR_ABSORPTION_COEFFS, SPEED_OF_SOUND,
};
