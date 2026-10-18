//! Poisson-disk sampling kernels for Percentage-Closer Filtering (PCF).

use korppi_math::Vec2;

/// 16-tap rotated Poisson-disk sampling kernel normalized to the unit disk (radius <= 1.0).
pub const POISSON_DISK_16: [Vec2; 16] = [
    Vec2::new(-0.76348, -0.32344),
    Vec2::new(0.76639, -0.62319),
    Vec2::new(-0.07633, -0.75326),
    Vec2::new(0.27958, 0.23818),
    Vec2::new(-0.74230, 0.37097),
    Vec2::new(-0.66091, -0.71253),
    Vec2::new(-0.31023, 0.22432),
    Vec2::new(0.79010, 0.61313),
    Vec2::new(0.35924, -0.79033),
    Vec2::new(0.43558, -0.38396),
    Vec2::new(-0.21476, -0.33954),
    Vec2::new(0.64189, 0.15472),
    Vec2::new(-0.19605, 0.80811),
    Vec2::new(-0.65982, 0.74110),
    Vec2::new(0.16202, 0.63738),
    Vec2::new(0.11657, -0.11428),
];
