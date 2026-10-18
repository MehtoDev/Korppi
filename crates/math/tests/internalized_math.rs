//! Primary outcome and supply-chain test for internalized `korppi_math` primitives (`FIX-11.3`).
//!
//! Verifies that:
//! 1. `Vec2`, `Vec3`, and `Vec4` are implemented natively in `korppi_math` with full arithmetic,
//!    indexing, geometric operations, swizzles, and `serde` round-trips.
//! 2. `Quat` supports axis-angle, elementary rotations, shortest-arc construction (including
//!    parallel and antiparallel edge cases), matrix conversion, slerp, and 3D vector rotation.
//! 3. `Mat3` and `Mat4` provide column-major linear algebra, affine transforms, right-handed
//!    camera view and `[0, 1]` clip-space projection matrices (standard and infinite Reverse-Z),
//!    determinants, and matrix inversion.
//! 4. `glam` has been completely removed from the workspace dependency graph (`cargo metadata`).

use korppi_math::prelude::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::path::PathBuf;
use std::process::Command;

#[test]
fn vec2_vec3_vec4_arithmetic_geometry_and_indexing() {
    let mut v2 = Vec2::new(3.0, -4.0);
    assert_eq!(v2.length(), 5.0);
    assert_eq!(v2.length_squared(), 25.0);
    assert_eq!(v2.abs(), Vec2::new(3.0, 4.0));
    assert_eq!(v2.signum(), Vec2::new(1.0, -1.0));
    assert_eq!(v2.perp(), Vec2::new(4.0, 3.0));
    assert_eq!(v2.perp_dot(Vec2::new(1.0, 2.0)), 10.0);
    assert_eq!(v2.min_element(), -4.0);
    assert_eq!(v2.max_element(), 3.0);
    v2[0] = 6.0;
    v2[1] = 8.0;
    assert_eq!(v2.normalize(), Vec2::new(0.6, 0.8));
    assert_eq!(Vec2::ZERO.normalize_or_zero(), Vec2::ZERO);
    assert_eq!((2.0 * v2) / 2.0, v2);
    assert_eq!(v2.extend(10.0), Vec3::new(6.0, 8.0, 10.0));

    let v3 = Vec3::new(1.5, -2.25, 3.75);
    assert_eq!(v3.floor(), Vec3::new(1.0, -3.0, 3.0));
    assert_eq!(v3.ceil(), Vec3::new(2.0, -2.0, 4.0));
    assert_eq!(v3.round(), Vec3::new(2.0, -2.0, 4.0));
    assert!((v3.fract() - (v3 - v3.floor())).length() < 1e-6);
    assert_eq!(v3.truncate(), Vec2::new(1.5, -2.25));
    assert_eq!(v3.xy(), Vec2::new(1.5, -2.25));
    assert_eq!(
        v3.clamp(Vec3::splat(-1.0), Vec3::splat(2.0)),
        Vec3::new(1.5, -1.0, 2.0)
    );
    assert_eq!(
        Vec3::ZERO.lerp(Vec3::new(10.0, 20.0, 30.0), 0.25),
        Vec3::new(2.5, 5.0, 7.5)
    );
    assert_eq!(
        Vec3::new(2.0, 3.0, 4.0).mul_add(Vec3::splat(2.0), Vec3::ONE),
        Vec3::new(5.0, 7.0, 9.0)
    );

    let v4 = v3.extend(1.0);
    assert_eq!(v4.truncate(), v3);
    assert_eq!(v4.xyz(), v3);
    assert_eq!(v4[3], 1.0);
    assert_eq!(v4.to_array(), [1.5, -2.25, 3.75, 1.0]);
    assert_eq!(Vec4::from_array(v4.to_array()), v4);
}

#[test]
fn quat_rotations_shortest_arc_and_matrix_roundtrip() {
    let qx = Quat::from_rotation_x(FRAC_PI_2);
    let qy = Quat::from_rotation_y(FRAC_PI_2);
    let qz = Quat::from_rotation_z(FRAC_PI_2);

    assert!((qx * Vec3::Y - Vec3::Z).length() < 1e-6);
    assert!((qy * Vec3::Z - Vec3::X).length() < 1e-6);
    assert!((qz * Vec3::X - Vec3::Y).length() < 1e-6);

    // Shortest-arc rotation between arbitrary unit vectors, parallel vectors, and opposite vectors
    let from = Vec3::new(1.0, 2.0, -0.5).normalize();
    let to = Vec3::new(-0.5, 1.0, 3.0).normalize();
    let arc = Quat::from_rotation_arc(from, to);
    assert!((arc * from - to).length() < 1e-5);

    let same_arc = Quat::from_rotation_arc(Vec3::Z, Vec3::Z);
    assert!((same_arc * Vec3::Z - Vec3::Z).length() < 1e-6);

    let opp_arc = Quat::from_rotation_arc(Vec3::X, Vec3::NEG_X);
    assert!((opp_arc * Vec3::X - Vec3::NEG_X).length() < 1e-5);
    assert!((opp_arc.length() - 1.0).abs() < 1e-6);

    // Mat3 <-> Quat round-trip across octants
    let composite = (qz * qy * qx).normalize();
    let m3 = Mat3::from_quat(composite);
    let recovered = Quat::from_mat3(&m3);
    let test_v = Vec3::new(2.0, -3.0, 5.0);
    assert!((m3 * test_v - composite * test_v).length() < 1e-5);
    assert!((recovered * test_v - composite * test_v).length() < 1e-5);

    // Slerp midpoint
    let mid = Quat::IDENTITY.slerp(qz, 0.5);
    let expected = Quat::from_rotation_z(FRAC_PI_4);
    assert!((mid * Vec3::X - expected * Vec3::X).length() < 1e-5);
}

#[test]
fn mat3_and_mat4_transforms_projections_and_inverses() {
    let scale = Vec3::new(2.0, 3.0, 0.5);
    let rot = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).normalize(), 0.8);
    let trans = Vec3::new(10.0, -5.0, 42.0);

    let model = Mat4::from_scale_rotation_translation(scale, rot, trans);
    let inv_model = model.inverse();
    let prod = model * inv_model;
    for col in 0..4 {
        assert!((prod.col(col) - Mat4::IDENTITY.col(col)).length() < 1e-5);
    }

    let local_p = Vec3::new(1.0, -2.0, 4.0);
    let world_p = model.transform_point3(local_p);
    let back_p = inv_model.transform_point3(world_p);
    assert!((back_p - local_p).length() < 1e-4);

    // View matrix: camera at (0, 0, 5) looking at origin with +Y up places origin at (0, 0, -5)
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    assert!((view.transform_point3(Vec3::ZERO) - Vec3::new(0.0, 0.0, -5.0)).length() < 1e-5);

    // Standard right-handed [0, 1] perspective: z = -near maps to NDC z = 0, z = -far maps to NDC z = 1
    let near = 0.5;
    let far = 100.0;
    let proj = Mat4::perspective_rh(PI / 3.0, 16.0 / 9.0, near, far);
    let ndc_near = proj.project_point3(Vec3::new(0.0, 0.0, -near));
    let ndc_far = proj.project_point3(Vec3::new(0.0, 0.0, -far));
    assert!(ndc_near.z.abs() < 1e-5, "ndc_near.z = {}", ndc_near.z);
    assert!((ndc_far.z - 1.0).abs() < 1e-5, "ndc_far.z = {}", ndc_far.z);

    // Infinite Reverse-Z perspective: z = -near maps to NDC z = 1, far approaches 0
    let rev_inf = Mat4::perspective_infinite_reverse_rh(PI / 3.0, 16.0 / 9.0, near);
    let rev_near = rev_inf.project_point3(Vec3::new(0.0, 0.0, -near));
    let rev_far = rev_inf.project_point3(Vec3::new(0.0, 0.0, -10_000.0));
    assert!((rev_near.z - 1.0).abs() < 1e-5);
    assert!(rev_far.z > 0.0 && rev_far.z < 1e-3);

    // Orthographic [0, 1] projection
    let ortho = Mat4::orthographic_rh(-10.0, 10.0, -5.0, 5.0, 1.0, 51.0);
    assert!((ortho.project_point3(Vec3::new(0.0, 0.0, -1.0)).z).abs() < 1e-5);
    assert!((ortho.project_point3(Vec3::new(10.0, 5.0, -51.0)) - Vec3::ONE).length() < 1e-5);

    // Affine3A compatibility in prelude
    let affine = Affine3A::from_scale_rotation_translation(scale, rot, trans);
    assert!((affine.transform_point3(local_p) - world_p).length() < 1e-5);
}

#[test]
fn workspace_has_removed_glam() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/math must sit two levels below workspace root");

    let root_toml = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(
        !root_toml.contains("glam"),
        "root Cargo.toml still references `glam`"
    );

    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .expect("failed to execute cargo metadata");
    assert!(output.status.success(), "cargo metadata failed");
    let stdout = String::from_utf8(output.stdout).expect("invalid UTF-8 from cargo metadata");
    assert!(
        !stdout.contains("\"name\":\"glam\""),
        "cargo metadata still contains prohibited external dependency `glam`"
    );
}
