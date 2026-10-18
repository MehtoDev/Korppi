//! Octahedral projection mapping for omnidirectional point light shadows.

use korppi_math::{Vec2, Vec3};

/// Helper sign function where zero maps to +1.0.
#[inline]
fn oct_sign(val: f32) -> f32 {
    if val >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Projects a 3D unit direction vector onto a 2D octahedral UV coordinate in $[0, 1]^2$.
pub fn encode_octahedral(direction: Vec3) -> Vec2 {
    let norm = direction.normalize();
    let l1_norm = norm.x.abs() + norm.y.abs() + norm.z.abs();

    if l1_norm < 1e-6 {
        return Vec2::splat(0.5);
    }

    let mut p = Vec2::new(norm.x / l1_norm, norm.y / l1_norm);

    if norm.z < 0.0 {
        let px = (1.0 - p.y.abs()) * oct_sign(p.x);
        let py = (1.0 - p.x.abs()) * oct_sign(p.y);
        p = Vec2::new(px, py);
    }

    p * 0.5 + Vec2::splat(0.5)
}

/// Unfolds a 2D octahedral UV coordinate in $[0, 1]^2$ back to a 3D unit direction vector.
pub fn decode_octahedral(uv: Vec2) -> Vec3 {
    let p = uv * 2.0 - Vec2::splat(1.0);
    let mut v = Vec3::new(p.x, p.y, 1.0 - p.x.abs() - p.y.abs());

    if v.z < 0.0 {
        let vx = (1.0 - p.y.abs()) * oct_sign(p.x);
        let vy = (1.0 - p.x.abs()) * oct_sign(p.y);
        v.x = vx;
        v.y = vy;
    }

    v.normalize()
}
