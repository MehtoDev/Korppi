//! Regression tests for Milestone FIX-13.1 (`MTH-01` through `MTH-05`).

use korppi_math::simd::{portable, FloatW};
use korppi_math::{Aabb, Aabb2d, Mat4, Quat, Ray, Sphere, Transform, Vec2, Vec3};
use serde::{de::DeserializeOwned, Serialize};
use std::f32::consts::FRAC_PI_2;

/// MTH-01 (Probe M1): `Transform::inverse` must rotate `-translation` before scaling by `inv_scale`,
/// and `Transform::inverse_transform_point` / `inverse_transform_vector` must invert arbitrary
/// non-uniform scale + rotation + translation transforms accurately.
#[test]
fn mth01_transform_inverse_under_rotation_and_scale() {
    let t = Transform {
        translation: Vec3::new(10.0, 20.0, 30.0),
        rotation: Quat::from_rotation_z(FRAC_PI_2),
        scale: Vec3::new(2.0, 4.0, 1.0),
    };

    // M1a: Even with non-uniform scale, `t.transform_point(t.inverse().transform_point(ZERO))`
    // must map the origin back to `Vec3::ZERO`.
    let roundtrip_zero = t.transform_point(t.inverse().transform_point(Vec3::ZERO));
    assert!(
        roundtrip_zero.abs_diff_eq(Vec3::ZERO, 1e-4),
        "expected ZERO, got {roundtrip_zero:?}"
    );

    // Uniform scale + rotation + translation: `Transform::inverse()` is an exact TRS inverse for all points.
    let t_uniform = Transform {
        translation: Vec3::new(-5.0, 12.0, 3.5),
        rotation: Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 1.1),
        scale: Vec3::splat(2.5),
    };
    let p = Vec3::new(3.0, -7.0, 4.0);
    let world_u = t_uniform.transform_point(p);
    let local_u = t_uniform.inverse().transform_point(world_u);
    assert!(
        local_u.abs_diff_eq(p, 1e-4),
        "uniform inverse roundtrip failed: expected {p:?}, got {local_u:?}"
    );

    // M1b: Non-uniform scale + rotation + translation: `inverse_transform_point` and
    // `inverse_transform_vector` must match `t.to_matrix().inverse()`.
    let world_p = t.transform_point(p);
    let recovered_p = t.inverse_transform_point(world_p);
    let mat_p = t.to_matrix().inverse().transform_point3(world_p);
    assert!(
        recovered_p.abs_diff_eq(p, 1e-4),
        "inverse_transform_point failed: expected {p:?}, got {recovered_p:?}"
    );
    assert!(recovered_p.abs_diff_eq(mat_p, 1e-4));

    let v = Vec3::new(-2.0, 5.0, 1.5);
    let world_v = t.transform_vector(v);
    let recovered_v = t.inverse_transform_vector(world_v);
    assert!(
        recovered_v.abs_diff_eq(v, 1e-4),
        "inverse_transform_vector failed: expected {v:?}, got {recovered_v:?}"
    );
}

/// MTH-02 (Probe M2): `Ray::new` with zero direction must not produce `NaN`, and
/// `Aabb::intersect_ray` must reject `NaN`/degenerate rays and handle boundary slabs cleanly.
#[test]
fn mth02_ray_new_zero_and_aabb_intersect_ray_nan_and_boundary_slabs() {
    // M2a: Ray::new(ZERO, ZERO) must not produce NaN.
    let zero_ray = Ray::new(Vec3::ZERO, Vec3::ZERO);
    assert!(
        zero_ray.direction.is_finite() && zero_ray.direction.is_normalized(),
        "Ray::new(ZERO, ZERO) must produce a finite unit fallback direction, got {:?}",
        zero_ray.direction
    );

    let distant = Aabb::new(Vec3::splat(1000.0), Vec3::splat(1001.0));
    assert_eq!(distant.intersect_ray(&zero_ray, 0.0, 100.0), None);

    // Directly constructed NaN or zero direction rays must return None, never Some(t_min).
    let nan_ray = Ray {
        origin: Vec3::ZERO,
        direction: Vec3::NAN,
    };
    assert_eq!(distant.intersect_ray(&nan_ray, 0.0, f32::INFINITY), None);

    let raw_zero_ray = Ray {
        origin: Vec3::ZERO,
        direction: Vec3::ZERO,
    };
    assert_eq!(
        distant.intersect_ray(&raw_zero_ray, 0.0, f32::INFINITY),
        None
    );

    // M2b: Ray parallel to X axis lying on the x = max face with -0.0 direction component.
    let box_on_face = Aabb::new(Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.0, 2.0, 1.0));
    let boundary_ray = Ray {
        origin: Vec3::new(1.0, 0.0, 0.0),
        direction: Vec3::new(-0.0, 1.0, 0.0),
    };
    assert_eq!(
        box_on_face.intersect_ray(&boundary_ray, 0.0, 10.0),
        Some(1.0)
    );

    // Ray parallel to X axis outside the X slab must miss.
    let outside_parallel = Ray {
        origin: Vec3::new(1.5, 0.0, 0.0),
        direction: Vec3::new(0.0, 1.0, 0.0),
    };
    assert_eq!(
        box_on_face.intersect_ray(&outside_parallel, 0.0, 10.0),
        None
    );

    // Empty AABB must never be hit.
    assert_eq!(
        Aabb::default().intersect_ray(&Ray::new(Vec3::ZERO, Vec3::X), 0.0, 100.0),
        None
    );
}

/// MTH-03 (Probe M3): Empty `Aabb` and `Aabb2d` must report `is_empty() == true`, zero area/surface_area/perimeter,
/// zero center/extents, and `from_min_max` / `new` must not invert empty bounds into `[-inf, +inf]`.
#[test]
fn mth03_empty_aabb_and_aabb2d_invariants() {
    let empty3 = Aabb::default();
    assert!(empty3.is_empty());
    assert_eq!(empty3, Aabb::empty());
    assert_eq!(empty3.surface_area(), 0.0);
    assert_eq!(empty3.center(), Vec3::ZERO);
    assert_eq!(empty3.extents(), Vec3::ZERO);
    assert_eq!(empty3.half_extents(), Vec3::ZERO);

    let rebuilt3 = Aabb::from_min_max(empty3.min, empty3.max);
    assert!(
        rebuilt3.is_empty(),
        "Aabb::from_min_max on empty bounds must remain empty, got {rebuilt3:?}"
    );

    let empty2 = Aabb2d::empty();
    assert!(empty2.is_empty());
    assert_eq!(empty2, Aabb2d::default());
    assert_eq!(empty2.area(), 0.0);
    assert_eq!(empty2.perimeter(), 0.0);
    assert_eq!(empty2.center(), Vec2::ZERO);
    assert_eq!(empty2.extents(), Vec2::ZERO);
    assert_eq!(empty2.half_extents(), Vec2::ZERO);

    let rebuilt2 = Aabb2d::new(empty2.min, empty2.max);
    assert!(
        rebuilt2.is_empty(),
        "Aabb2d::new on empty bounds must remain empty, got {rebuilt2:?}"
    );
}

/// MTH-04 (Probe M4): `portable::FloatW` must match `FloatW` bit-for-bit on NaN-generating and
/// NaN-propagating arithmetic (`0.0 / 0.0`, `inf - inf`, `0.0 * inf`, `sqrt(-1.0)`).
#[test]
fn mth04_simd_floatw_nan_bits_match_sse2_exactly() {
    let a = [0.0f32, 0.0, f32::NAN, f32::INFINITY];
    let b = [0.0f32, -0.0, 1.0, f32::INFINITY];
    let na = FloatW::from_array(a);
    let nb = FloatW::from_array(b);
    let pa = portable::FloatW::from_array(a);
    let pb = portable::FloatW::from_array(b);

    let div_n = (na / nb).to_array().map(f32::to_bits);
    let div_p = (pa / pb).to_array().map(f32::to_bits);
    assert_eq!(div_n, div_p, "division NaN bits must match bit-for-bit");

    let sub_n = (na - nb).to_array().map(f32::to_bits);
    let sub_p = (pa - pb).to_array().map(f32::to_bits);
    assert_eq!(sub_n, sub_p, "subtraction NaN bits must match bit-for-bit");

    let mul_a = [0.0f32, f32::NEG_INFINITY, f32::NAN, -1.0];
    let mul_b = [f32::INFINITY, 0.0, -0.0, f32::NAN];
    let mna = FloatW::from_array(mul_a);
    let mnb = FloatW::from_array(mul_b);
    let mpa = portable::FloatW::from_array(mul_a);
    let mpb = portable::FloatW::from_array(mul_b);
    assert_eq!(
        (mna * mnb).to_array().map(f32::to_bits),
        (mpa * mpb).to_array().map(f32::to_bits),
        "multiplication NaN bits must match bit-for-bit"
    );
    assert_eq!(
        mna.sqrt().to_array().map(f32::to_bits),
        mpa.sqrt().to_array().map(f32::to_bits),
        "sqrt NaN bits must match bit-for-bit"
    );
}

fn assert_serde_bounds<T: Serialize + DeserializeOwned>() {}

/// MTH-05: `Transform`, `Ray`, and `Sphere` implement `Serialize` and `Deserialize`, and
/// `Transform::compute_matrix`, `Transform::look_at`, and `Mat4::to_scale_rotation_translation` work.
#[test]
fn mth05_serde_bounds_and_transform_matrix_helpers() {
    assert_serde_bounds::<Transform>();
    assert_serde_bounds::<Ray>();
    assert_serde_bounds::<Sphere>();

    let mut t = Transform::from_xyz(0.0, 0.0, 5.0);
    t.look_at(Vec3::ZERO, Vec3::Y);
    assert!(
        t.forward().abs_diff_eq(Vec3::NEG_Z, 1e-5),
        "look_at from (0,0,5) to origin should face -Z, got {:?}",
        t.forward()
    );
    assert_eq!(t.compute_matrix(), t.to_matrix());

    let expected_scale = Vec3::new(2.0, 3.0, 4.0);
    let expected_rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.7);
    let expected_trans = Vec3::new(-4.0, 5.0, 6.0);
    let mat = Mat4::from_scale_rotation_translation(expected_scale, expected_rot, expected_trans);
    let (s, r, tr) = mat.to_scale_rotation_translation();
    assert!(s.abs_diff_eq(expected_scale, 1e-5));
    assert!(r.abs_diff_eq(expected_rot, 1e-5) || (-r).abs_diff_eq(expected_rot, 1e-5));
    assert!(tr.abs_diff_eq(expected_trans, 1e-5));
}
