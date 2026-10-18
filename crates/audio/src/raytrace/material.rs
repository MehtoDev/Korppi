//! Acoustic material definition with 3-band absorption, scattering, and transmission properties.

use korppi_ecs::Component;
use serde::{Deserialize, Serialize};

/// 3-band acoustic material modeling frequency-dependent surface interaction.
///
/// Frequency bands are defined according to ISO 354:
/// - Low: $250 - 500\text{ Hz}$ (Bass resonance)
/// - Mid: $1 - 2\text{ kHz}$ (Speech, ear sensitivity peak)
/// - High: $8 - 16\text{ kHz}$ (Treble, crisp reflections)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcousticMaterial {
    /// Absorption coefficient $\alpha(b) \in [0.0, 1.0]$ per frequency band.
    pub absorption: [f32; 3],
    /// Scattering coefficient $s \in [0.0, 1.0]$ (diffuse vs. specular reflection).
    pub scattering: f32,
    /// Transmission coefficient $\tau(b) \in [0.0, 1.0]$ per frequency band through barriers.
    pub transmission: [f32; 3],
}

impl Component for AcousticMaterial {}

impl Default for AcousticMaterial {
    fn default() -> Self {
        Self::generic()
    }
}

impl AcousticMaterial {
    /// Creates a custom acoustic material.
    #[must_use]
    pub const fn new(absorption: [f32; 3], scattering: f32, transmission: [f32; 3]) -> Self {
        Self {
            absorption,
            scattering,
            transmission,
        }
    }

    /// Generic architectural surface baseline.
    #[must_use]
    pub const fn generic() -> Self {
        Self {
            absorption: [0.10, 0.20, 0.30],
            scattering: 0.10,
            transmission: [0.100, 0.050, 0.020],
        }
    }

    /// Smooth concrete / stone masonry (highly reverberant, low absorption).
    #[must_use]
    pub const fn concrete() -> Self {
        Self {
            absorption: [0.02, 0.03, 0.05],
            scattering: 0.05,
            transmission: [0.010, 0.005, 0.001],
        }
    }

    /// Brick wall (solid reflector with slight surface roughness).
    #[must_use]
    pub const fn brick() -> Self {
        Self {
            absorption: [0.03, 0.04, 0.07],
            scattering: 0.15,
            transmission: [0.015, 0.008, 0.002],
        }
    }

    /// Hardwood floor or wall paneling.
    #[must_use]
    pub const fn hardwood() -> Self {
        Self {
            absorption: [0.15, 0.10, 0.08],
            scattering: 0.10,
            transmission: [0.080, 0.030, 0.010],
        }
    }

    /// Heavy carpet on concrete pad (absorbs high frequencies heavily).
    #[must_use]
    pub const fn carpet() -> Self {
        Self {
            absorption: [0.05, 0.30, 0.65],
            scattering: 0.40,
            transmission: [0.010, 0.005, 0.001],
        }
    }

    /// Drywall / interior plaster partition (porous transmission, resonant bass absorber).
    #[must_use]
    pub const fn drywall() -> Self {
        Self {
            absorption: [0.28, 0.12, 0.10],
            scattering: 0.10,
            transmission: [0.150, 0.080, 0.030],
        }
    }

    /// Glass window pane (specular high-frequency reflector).
    #[must_use]
    pub const fn glass() -> Self {
        Self {
            absorption: [0.18, 0.06, 0.04],
            scattering: 0.05,
            transmission: [0.080, 0.040, 0.015],
        }
    }

    /// Studio acoustic foam (dead acoustic space, high absorption across all bands).
    #[must_use]
    pub const fn acoustic_foam() -> Self {
        Self {
            absorption: [0.20, 0.75, 0.95],
            scattering: 0.50,
            transmission: [0.005, 0.002, 0.001],
        }
    }

    /// Sheet metal panel (bright specular metallic reflector).
    #[must_use]
    pub const fn sheet_metal() -> Self {
        Self {
            absorption: [0.20, 0.08, 0.05],
            scattering: 0.08,
            transmission: [0.030, 0.015, 0.005],
        }
    }

    /// Computes the reflection coefficient $R(b) = 1.0 - \alpha(b)$ for each band.
    #[must_use]
    pub fn reflection(&self) -> [f32; 3] {
        [
            (1.0 - self.absorption[0]).clamp(0.0, 1.0),
            (1.0 - self.absorption[1]).clamp(0.0, 1.0),
            (1.0 - self.absorption[2]).clamp(0.0, 1.0),
        ]
    }

    /// Computes the average absorption across all 3 bands $\bar{\alpha}$.
    #[must_use]
    pub fn average_absorption(&self) -> f32 {
        (self.absorption[0] + self.absorption[1] + self.absorption[2]) / 3.0
    }
}
