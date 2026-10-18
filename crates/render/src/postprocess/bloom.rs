//! Jorge Jimenez dual-filtering progressive bloom downsample/upsample pyramid (`RND-15`).

use korppi_math::Vec3;

/// Configuration parameters for the progressive HDR bloom pyramid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomConfig {
    /// Luminance threshold where pixels start bleeding into the bloom chain.
    pub threshold: f32,
    /// Soft-knee parameter ($\in [0.0, 1.0]$) smoothing the transition at threshold.
    pub soft_knee: f32,
    /// Overall bloom contribution intensity blended onto the scene.
    pub intensity: f32,
    /// Upsample filter kernel radius multiplier.
    pub filter_radius: f32,
    /// Total number of downsample/upsample mip pyramid passes (typically 5 to 7).
    pub mip_levels: u32,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            soft_knee: 0.5,
            intensity: 0.05,
            filter_radius: 1.0,
            mip_levels: 5,
        }
    }
}

/// Applies a soft-knee threshold curve to isolate HDR highlights for bloom.
///
/// Smoothly transitions from zero to full linear brightness around the knee zone:
/// $$[\text{threshold} - \text{knee}, \text{threshold} + \text{knee}]$$
pub fn bloom_soft_knee(color: Vec3, threshold: f32, soft_knee: f32) -> Vec3 {
    let c = color.max(Vec3::ZERO);
    let l = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;

    let knee = (threshold * soft_knee.clamp(0.0, 1.0)).max(1e-5);
    let diff = l - threshold + knee;
    let rq = diff.clamp(0.0, 2.0 * knee);
    let soft = (rq * rq) / (4.0 * knee + 1e-5);

    let weight = (l - threshold).max(soft) / l.max(1e-5);
    c * weight.clamp(0.0, 1.0)
}

/// Computes Karis partial average weight for suppressing subpixel specular fireflies.
///
/// Uses reciprocal luminance weighting:
/// $$w = \frac{1}{1 + L}$$
#[inline]
pub fn compute_karis_average_weight(color: Vec3) -> f32 {
    let c = color.max(Vec3::ZERO);
    let l = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    1.0 / (1.0 + l)
}

/// 13-tap Jorge Jimenez downsample filter weights (`RND-15`).
///
/// Combines 5 overlapping 4-tap bilinear boxes (1 inner box with weight 0.5 and 4 outer corner boxes
/// with weight 0.125 each), summing to 1.0:
/// - 4 inner diagonal box samples: $0.5 / 4 = 0.125$ each
/// - 4 outer corner samples: $0.125 / 4 = 0.03125$ each
/// - 4 cross edge samples (shared by 2 outer boxes): $2 \times 0.03125 = 0.0625$ each
/// - 1 center sample (shared by all 4 outer boxes): $4 \times 0.03125 = 0.125$
pub const DOWNSAMPLE_13TAP_WEIGHTS: [f32; 13] = [
    // 4 inner box samples (0.5 total / 4 = 0.125 each)
    0.125, 0.125, 0.125, 0.125, // 4 outer corner samples (0.125 total / 4 = 0.03125 each)
    0.03125, 0.03125, 0.03125, 0.03125, // 4 cross edge samples (0.25 total / 4 = 0.0625 each)
    0.0625, 0.0625, 0.0625, 0.0625, // 1 center point (shared by all 4 outer boxes = 0.125)
    0.125,
];

/// 9-tap 3x3 tent upsample filter weights.
///
/// Normalized weights for bilinear upsampling interpolation:
/// Center: 4/16 (0.25), Edges: 2/16 (0.125) each, Corners: 1/16 (0.0625) each.
pub const UPSAMPLE_9TAP_WEIGHTS: [f32; 9] = [
    0.0625, 0.125, 0.0625, // top row
    0.125, 0.25, 0.125, // middle row
    0.0625, 0.125, 0.0625, // bottom row
];
