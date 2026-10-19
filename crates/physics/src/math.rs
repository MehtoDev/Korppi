//! Box3D's matrix helpers (`math_functions.h`), with its order of float operations, so that
//! ported code rounds as Box3D does. Matrices are column-major like Box3D's `b3Matrix3`
//! (`cx`, `cy`, `cz` are `Mat3`'s `x_axis`, `y_axis`, `z_axis`).

use korppi_math::{Mat3, Quat, Vec3};

/// Box3D's smallest determinant treated as invertible, `1000 * FLT_MIN`.
const MIN_DETERMINANT: f32 = 1000.0 * f32::MIN_POSITIVE;

/// Box3D `b3Det`: `dot(cx, cross(cy, cz))`.
pub(crate) fn det(m: &Mat3) -> f32 {
    m.x_axis.dot(m.y_axis.cross(m.z_axis))
}

/// The cofactor columns of Box3D's inverses, scaled by `1 / det`, or `None` if `m` is
/// singular.
fn scaled_cofactors(m: &Mat3) -> Option<Mat3> {
    let det = det(m);
    if det.abs() > MIN_DETERMINANT {
        let inv_det = 1.0 / det;
        Some(Mat3::from_cols(
            m.y_axis.cross(m.z_axis) * inv_det,
            m.z_axis.cross(m.x_axis) * inv_det,
            m.x_axis.cross(m.y_axis) * inv_det,
        ))
    } else {
        None
    }
}

/// Box3D `b3InvertMatrix`: the inverse, or zero if `m` is singular.
pub(crate) fn invert(m: &Mat3) -> Mat3 {
    scaled_cofactors(m).map_or(Mat3::ZERO, |c| c.transpose())
}

/// Box3D `b3InvertT`: the inverse transposed, which is the inverse of a symmetric matrix, or
/// zero if `m` is singular.
pub(crate) fn invert_symmetric(m: &Mat3) -> Mat3 {
    scaled_cofactors(m).unwrap_or(Mat3::ZERO)
}

/// Box3D `b3Solve3`: `x` with `m x = a`, or zero if `m` is singular.
pub(crate) fn solve3(m: &Mat3, a: Vec3) -> Vec3 {
    let det = det(m);
    if det.abs() > MIN_DETERMINANT {
        let inv_det = 1.0 / det;
        let s = Mat3::from_cols(
            m.y_axis.cross(m.z_axis),
            m.z_axis.cross(m.x_axis),
            m.x_axis.cross(m.y_axis),
        );
        Vec3::new(
            inv_det * s.x_axis.dot(a),
            inv_det * s.y_axis.dot(a),
            inv_det * s.z_axis.dot(a),
        )
    } else {
        Vec3::ZERO
    }
}

/// Box3D `b3Solve2` for the symmetric matrix `[[xx, xy], [xy, yy]]`: `x` with `m x = b`, or
/// zero unless the determinant is positive.
pub(crate) fn solve2((xx, xy, yy): (f32, f32, f32), b: (f32, f32)) -> (f32, f32) {
    let det = xx * yy - xy * xy;
    if det > MIN_DETERMINANT {
        let inv_det = 1.0 / det;
        (
            inv_det * yy * b.0 - inv_det * xy * b.1,
            -inv_det * xy * b.0 + inv_det * xx * b.1,
        )
    } else {
        (0.0, 0.0)
    }
}

/// Box3D `b3Skew`: the matrix of `cross(v, _)`.
pub(crate) fn skew(v: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, v.z, -v.y),
        Vec3::new(-v.z, 0.0, v.x),
        Vec3::new(v.y, -v.x, 0.0),
    )
}

/// The effective mass matrix of a point-to-point constraint at anchors `r_a` and `r_b`
/// (Box3D: `(mA + mB) E - skew(rA) iA skew(rA) - skew(rB) iB skew(rB)`).
pub(crate) fn point_mass(m: (f32, f32), i_a: &Mat3, i_b: &Mat3, r_a: Vec3, r_b: Vec3) -> Mat3 {
    let s_a = skew(r_a);
    let s_b = skew(r_b);
    let k_a = s_a * (*i_a * s_a);
    let k_b = s_b * (*i_b * s_b);
    let mut k = -(k_a + k_b);
    k.x_axis.x += m.0 + m.1;
    k.y_axis.y += m.0 + m.1;
    k.z_axis.z += m.0 + m.1;
    k
}

/// Box3D `b3Atan2`: a minimax polynomial approximation of `atan2(y, x)`, the same on every
/// platform (`f32::atan2` depends on the C library).
pub(crate) fn atan2(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    let (ax, ay) = (x.abs(), y.abs());
    let mx = ay.max(ax);
    let mn = ay.min(ax);
    let a = mn / mx;
    // Minimax polynomial approximation to atan(a) on [0, 1].
    let s = a * a;
    let c = s * a;
    let q = s * s;
    let mut r = 0.024_840_285 * q + 0.186_814_18;
    let t = -0.094_097_95 * q - 0.332_130_72;
    r = r * s + t;
    r = r * c + a;
    // Map to the full circle.
    if ay > ax {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = std::f32::consts::PI - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

/// Box3D `b3GetTwistAngle`: the twist of `q` about its z-axis, in [-π, π], whatever the
/// quaternion's sign.
pub(crate) fn twist_angle(q: Quat) -> f32 {
    let twist = if q.w < 0.0 {
        atan2(-q.z, -q.w)
    } else {
        atan2(q.z, q.w)
    };
    2.0 * twist
}

/// Box3D `b3GetSwingAngle`: how far `q` tilts its z-axis, in [0, π].
pub(crate) fn swing_angle(q: Quat) -> f32 {
    let x = (q.z * q.z + q.w * q.w).sqrt();
    let y = (q.x * q.x + q.y * q.y).sqrt();
    2.0 * atan2(y, x)
}

/// Box3D `b3GetQuatAngle`: the rotation angle of `q`.
pub(crate) fn quat_angle(q: Quat) -> f32 {
    let length = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
    2.0 * atan2(length, q.w)
}

/// Box3D `b3DeltaQuatToRotation`: the pseudo angular velocity `2 (target - q) conj(q)` that
/// turns `q` towards `target`, taking the shorter way.
pub(crate) fn delta_quat_to_rotation(q: Quat, target: Quat) -> Vec3 {
    let s = if q.dot(target) < 0.0 { -q } else { q };
    let diff = Quat::from_xyzw(
        target.x - s.x,
        target.y - s.y,
        target.z - s.z,
        target.w - s.w,
    );
    let product = diff * s.conjugate();
    Vec3::new(product.x, product.y, product.z) * 2.0
}

/// Box3D `b3RotateVector`: `v + 2 cross(q.v, cross(q.v, v) + q.s v)`.
pub(crate) fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let qv = Vec3::new(q.x, q.y, q.z);
    let t1 = qv.cross(v);
    let t2 = t1 + v * q.w;
    let t3 = qv.cross(t2);
    v + t3 * 2.0
}

/// Box3D `b3InvRotateVector`: `v + 2 cross(q.v, cross(q.v, v) - q.s v)`.
pub(crate) fn inv_rotate(q: Quat, v: Vec3) -> Vec3 {
    let qv = Vec3::new(q.x, q.y, q.z);
    let t1 = qv.cross(v);
    let t2 = t1 - v * q.w;
    let t3 = qv.cross(t2);
    v + t3 * 2.0
}

/// Box3D `b3InvMulQuat`: `conj(q1) q2`.
pub(crate) fn inv_mul_quat(q1: Quat, q2: Quat) -> Quat {
    let v1 = Vec3::new(q1.x, q1.y, q1.z);
    let v2 = Vec3::new(q2.x, q2.y, q2.z);
    let t1 = v2.cross(v1);
    let t2 = t1 + v2 * q1.w;
    let t3 = t2 - v1 * q2.w;
    Quat::from_xyzw(t3.x, t3.y, t3.z, q1.w * q2.w + v1.dot(v2))
}

/// Box3D `b3NormalizeQuat`: the unit quaternion, or the identity for a tiny one.
pub(crate) fn normalize_quat(q: Quat) -> Quat {
    let length_squared = q.dot(q);
    if length_squared > MIN_DETERMINANT {
        let s = 1.0 / length_squared.sqrt();
        Quat::from_xyzw(s * q.x, s * q.y, s * q.z, s * q.w)
    } else {
        Quat::IDENTITY
    }
}

/// Box3D `b3NLerp`: the normalized blend of `q1` and `q2`, the shorter way.
pub(crate) fn nlerp(q1: Quat, q2: Quat, alpha: f32) -> Quat {
    let q1 = if q1.dot(q2) < 0.0 { -q1 } else { q1 };
    let blend = |a: f32, b: f32| (1.0 - alpha) * a + alpha * b;
    normalize_quat(Quat::from_xyzw(
        blend(q1.x, q2.x),
        blend(q1.y, q2.y),
        blend(q1.z, q2.z),
        blend(q1.w, q2.w),
    ))
}

/// Box3D `b3Lerp`.
pub(crate) fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3::new(
        (1.0 - t) * a.x + t * b.x,
        (1.0 - t) * a.y + t * b.y,
        (1.0 - t) * a.z + t * b.z,
    )
}

/// Box3D `b3Steiner`: the inertia of `mass` at `offset` from a point, about that point (the
/// parallel axis theorem).
pub(crate) fn steiner(mass: f32, o: Vec3) -> Mat3 {
    let (ixx, iyy, izz) = (
        mass * (o.y * o.y + o.z * o.z),
        mass * (o.x * o.x + o.z * o.z),
        mass * (o.x * o.x + o.y * o.y),
    );
    let (ixy, ixz, iyz) = (-mass * o.x * o.y, -mass * o.x * o.z, -mass * o.y * o.z);
    Mat3::from_cols(
        Vec3::new(ixx, ixy, ixz),
        Vec3::new(ixy, iyy, iyz),
        Vec3::new(ixz, iyz, izz),
    )
}

/// Box3D `b3ModifiedCross`: the cross product with every term added, a bound on the speed of a
/// box corner `b` turning at `a`.
pub(crate) fn modified_cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z + a.z * b.y,
        a.z * b.x + a.x * b.z,
        a.x * b.y + a.y * b.x,
    )
}

/// A rigid transform (Box3D `b3Transform`): a rotation, then a translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Translation position.
    pub p: Vec3,
    /// Orientation quaternion.
    pub q: Quat,
}

/// Type alias for [`Transform`].
pub type Transform3d = Transform;

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<(Vec3, Quat)> for Transform {
    fn from((p, q): (Vec3, Quat)) -> Self {
        Self { p, q }
    }
}

impl Transform {
    /// The identity transform with zero translation and identity rotation.
    pub const IDENTITY: Self = Self {
        p: Vec3::ZERO,
        q: Quat::IDENTITY,
    };

    /// Constructs a transform from translation `p` and rotation `q`.
    #[must_use]
    pub const fn new(p: Vec3, q: Quat) -> Self {
        Self { p, q }
    }

    /// Box3D `b3TransformPoint`.
    #[must_use]
    pub fn apply(&self, v: Vec3) -> Vec3 {
        rotate(self.q, v) + self.p
    }

    /// Box3D `b3InvTransformPoint`.
    #[must_use]
    pub fn apply_inverse(&self, v: Vec3) -> Vec3 {
        inv_rotate(self.q, v - self.p)
    }

    /// Box3D `b3InvMulTransforms`: `other` in this transform's frame.
    #[must_use]
    pub fn inv_mul(&self, other: &Self) -> Self {
        Self::new(
            inv_rotate(self.q, other.p - self.p),
            inv_mul_quat(self.q, other.q),
        )
    }

    /// Box3D `b3MulTransforms`: compound transform `self * other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        Self::new(self.apply(other.p), self.q * other.q)
    }

    /// Box3D `b3InvertTransform`.
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self::new(inv_rotate(self.q, -self.p), self.q.conjugate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_follow_box3d() {
        let q = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.3);
        let v = Vec3::new(1.0, -2.0, 0.5);
        assert!((rotate(q, v) - q * v).length() < 1e-6);
        assert!((inv_rotate(q, v) - q.conjugate() * v).length() < 1e-6);
        let a = Transform::new(Vec3::new(1.0, 2.0, 3.0), q);
        let b = Transform::new(Vec3::new(-1.0, 0.5, 2.0), Quat::from_rotation_z(0.4));
        let b_in_a = a.inv_mul(&b);
        assert!((a.apply(b_in_a.apply(v)) - b.apply(v)).length() < 1e-5);
        assert!((a.inverse().apply(a.apply(v)) - v).length() < 1e-5);
        assert!((a.apply_inverse(a.apply(v)) - v).length() < 1e-5);
    }

    #[test]
    fn inverses_undo_the_matrix() {
        let m = Mat3::from_cols(
            Vec3::new(2.0, 0.5, 0.0),
            Vec3::new(0.5, 3.0, 0.25),
            Vec3::new(0.0, 0.25, 4.0),
        );
        for inverse in [invert(&m), invert_symmetric(&m)] {
            let identity = inverse * m;
            assert!(identity.abs_diff_eq(Mat3::IDENTITY, 1e-6), "{identity:?}");
        }
        let x = solve3(&m, Vec3::new(1.0, 2.0, 3.0));
        assert!((m * x - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
        assert_eq!(invert(&Mat3::ZERO), Mat3::ZERO);
        assert_eq!(solve3(&Mat3::ZERO, Vec3::ONE), Vec3::ZERO);
        let (x, y) = solve2((2.0, 0.5, 3.0), (1.0, 2.0));
        assert!((2.0 * x + 0.5 * y - 1.0).abs() < 1e-6 && (0.5 * x + 3.0 * y - 2.0).abs() < 1e-6);
    }

    #[test]
    fn angles_follow_box3d() {
        // Box3D's polynomial is within 2.7e-5 rad of atan2; the half-angle forms double that.
        for angle in [-3.0f32, -1.0, -0.1, 0.0, 0.2, 1.5, 3.0] {
            assert!(
                (atan2(angle.sin(), angle.cos()) - angle).abs() < 3e-5,
                "{angle}"
            );
            let q = Quat::from_rotation_z(angle);
            assert!((twist_angle(q) - angle).abs() < 6e-5);
            assert!(
                (twist_angle(-q) - angle).abs() < 6e-5,
                "the sign of q does not matter"
            );
            assert!((swing_angle(Quat::from_rotation_x(angle)) - angle.abs()).abs() < 6e-5);
            assert!((quat_angle(Quat::from_rotation_y(angle)) - angle.abs()).abs() < 6e-5);
        }
        let w = delta_quat_to_rotation(Quat::IDENTITY, Quat::from_rotation_z(0.01));
        assert!((w - Vec3::new(0.0, 0.0, 0.01)).length() < 1e-5, "{w:?}");
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(skew(v) * Vec3::X, v.cross(Vec3::X));
    }
}
