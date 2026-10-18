//! Decal representations and view-space Oriented Bounding Box (OBB) intersection tests.

use korppi_math::{Mat3, Quat, Vec3};

/// Un-deferred forward decal volume in view space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decal {
    /// View-space center position of the decal box.
    pub position_vs: Vec3,
    /// Half-extents along local decal axes.
    pub half_extents: Vec3,
    /// Orientation quaternion in view space.
    pub orientation_vs: Quat,
}

impl Decal {
    /// Creates a new oriented decal in view space.
    #[inline]
    pub fn new(position_vs: Vec3, half_extents: Vec3, orientation_vs: Quat) -> Self {
        Self {
            position_vs,
            half_extents,
            orientation_vs,
        }
    }

    /// Creates an axis-aligned decal in view space.
    #[inline]
    pub fn axis_aligned(position_vs: Vec3, half_extents: Vec3) -> Self {
        Self {
            position_vs,
            half_extents,
            orientation_vs: Quat::IDENTITY,
        }
    }
}

/// Tests intersection between a view-space decal OBB and a froxel AABB using the Separating Axis Theorem (SAT).
pub fn test_decal_vs_aabb(decal: &Decal, aabb_min: Vec3, aabb_max: Vec3) -> bool {
    let aabb_center = (aabb_min + aabb_max) * 0.5;
    let aabb_extents = (aabb_max - aabb_min) * 0.5;
    let t = decal.position_vs - aabb_center;

    let r_mat = Mat3::from_quat(decal.orientation_vs);
    let b0 = r_mat.col(0);
    let b1 = r_mat.col(1);
    let b2 = r_mat.col(2);

    let e_b = decal.half_extents;

    // Absolute rotation matrix elements with small epsilon to prevent degeneracy
    let r00 = b0.x.abs() + 1e-6;
    let r01 = b1.x.abs() + 1e-6;
    let r02 = b2.x.abs() + 1e-6;
    let r10 = b0.y.abs() + 1e-6;
    let r11 = b1.y.abs() + 1e-6;
    let r12 = b2.y.abs() + 1e-6;
    let r20 = b0.z.abs() + 1e-6;
    let r21 = b1.z.abs() + 1e-6;
    let r22 = b2.z.abs() + 1e-6;

    // 1. AABB face axes: A0 = (1, 0, 0), A1 = (0, 1, 0), A2 = (0, 0, 1)
    if t.x.abs() > aabb_extents.x + e_b.x * r00 + e_b.y * r01 + e_b.z * r02 {
        return false;
    }
    if t.y.abs() > aabb_extents.y + e_b.x * r10 + e_b.y * r11 + e_b.z * r12 {
        return false;
    }
    if t.z.abs() > aabb_extents.z + e_b.x * r20 + e_b.y * r21 + e_b.z * r22 {
        return false;
    }

    // 2. OBB face axes: B0, B1, B2
    if t.dot(b0).abs() > aabb_extents.x * r00 + aabb_extents.y * r10 + aabb_extents.z * r20 + e_b.x
    {
        return false;
    }
    if t.dot(b1).abs() > aabb_extents.x * r01 + aabb_extents.y * r11 + aabb_extents.z * r21 + e_b.y
    {
        return false;
    }
    if t.dot(b2).abs() > aabb_extents.x * r02 + aabb_extents.y * r12 + aabb_extents.z * r22 + e_b.z
    {
        return false;
    }

    // 3. 9 cross product axes: A_i x B_j
    // A0 x B0
    if (t.z * b0.y - t.y * b0.z).abs()
        > aabb_extents.y * r20 + aabb_extents.z * r10 + e_b.y * r02 + e_b.z * r01
    {
        return false;
    }
    // A0 x B1
    if (t.z * b1.y - t.y * b1.z).abs()
        > aabb_extents.y * r21 + aabb_extents.z * r11 + e_b.x * r02 + e_b.z * r00
    {
        return false;
    }
    // A0 x B2
    if (t.z * b2.y - t.y * b2.z).abs()
        > aabb_extents.y * r22 + aabb_extents.z * r12 + e_b.x * r01 + e_b.y * r00
    {
        return false;
    }

    // A1 x B0
    if (t.x * b0.z - t.z * b0.x).abs()
        > aabb_extents.x * r20 + aabb_extents.z * r00 + e_b.y * r12 + e_b.z * r11
    {
        return false;
    }
    // A1 x B1
    if (t.x * b1.z - t.z * b1.x).abs()
        > aabb_extents.x * r21 + aabb_extents.z * r01 + e_b.x * r12 + e_b.z * r10
    {
        return false;
    }
    // A1 x B2
    if (t.x * b2.z - t.z * b2.x).abs()
        > aabb_extents.x * r22 + aabb_extents.z * r02 + e_b.x * r11 + e_b.y * r10
    {
        return false;
    }

    // A2 x B0
    if (t.y * b0.x - t.x * b0.y).abs()
        > aabb_extents.x * r10 + aabb_extents.y * r00 + e_b.y * r22 + e_b.z * r21
    {
        return false;
    }
    // A2 x B1
    if (t.y * b1.x - t.x * b1.y).abs()
        > aabb_extents.x * r11 + aabb_extents.y * r01 + e_b.x * r22 + e_b.z * r20
    {
        return false;
    }
    // A2 x B2
    if (t.y * b2.x - t.x * b2.y).abs()
        > aabb_extents.x * r12 + aabb_extents.y * r02 + e_b.x * r21 + e_b.y * r20
    {
        return false;
    }

    true
}
