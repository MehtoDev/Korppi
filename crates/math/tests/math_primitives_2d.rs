//! Integration tests for 2D math primitives (Aabb2d, Ray2d, Rot2, Transform2d).

use korppi_math::{unwind_angle, Aabb2d, Ray2d, Rot2, Transform2d, Vec2};
use std::f32::consts::{FRAC_PI_2, PI};

#[test]
fn rot2_rotates_vectors_and_combines_angles() {
    let identity = Rot2::IDENTITY;
    assert_eq!(identity.c, 1.0);
    assert_eq!(identity.s, 0.0);
    assert_eq!(identity.angle(), 0.0);

    let v = Vec2::new(1.0, 0.0);
    assert_eq!(identity.rotate(v), v);

    // 90 degrees rotation counter-clockwise
    let rot90 = Rot2::from_angle(FRAC_PI_2);
    let rotated90 = rot90.rotate(v);
    assert!((rotated90 - Vec2::new(0.0, 1.0)).length() < 1e-5);
    assert!((rot90.angle() - FRAC_PI_2).abs() < 1e-5);

    // Inverse rotation
    let inv_rotated = rot90.inv_rotate(rotated90);
    assert!((inv_rotated - v).length() < 1e-5);
    let inv = rot90.inverse();
    assert!((inv.rotate(rotated90) - v).length() < 1e-5);

    // Multiplication combines rotations: 90 deg + 90 deg = 180 deg (PI)
    let rot180 = rot90 * rot90;
    let rotated180 = rot180.rotate(v);
    assert!((rotated180 - Vec2::new(-1.0, 0.0)).length() < 1e-5);
    assert!((rot180.angle().abs() - PI).abs() < 1e-5);
}

#[test]
fn transform2d_transforms_points_vectors_and_combines() {
    let translation = Vec2::new(10.0, 20.0);
    let rot90 = Rot2::from_angle(FRAC_PI_2);
    let xf = Transform2d::from_translation_rotation(translation, rot90);

    let point = Vec2::new(1.0, 0.0);
    // Transformed point: rotated by 90 deg -> (0.0, 1.0), then translated -> (10.0, 21.0)
    let transformed_point = xf.transform_point(point);
    assert!((transformed_point - Vec2::new(10.0, 21.0)).length() < 1e-5);

    // Inverse transformed point
    let inv_point = xf.inv_transform_point(transformed_point);
    assert!((inv_point - point).length() < 1e-5);

    // Transformed vector (only rotated, no translation)
    let vector = Vec2::new(2.0, 0.0);
    let transformed_vec = xf.transform_vector(vector);
    assert!((transformed_vec - Vec2::new(0.0, 2.0)).length() < 1e-5);
    let inv_vec = xf.inv_transform_vector(transformed_vec);
    assert!((inv_vec - vector).length() < 1e-5);

    // Combine parent and child transforms
    let child = Transform2d::from_translation(Vec2::new(5.0, 0.0));
    let combined = xf.combine(&child);
    // Child origin (5, 0) transformed by xf: rotated to (0, 5), plus (10, 20) -> (10, 25)
    assert!((combined.translation - Vec2::new(10.0, 25.0)).length() < 1e-5);
    assert_eq!(combined.rotation, rot90);
}

#[test]
fn transform2d_inv_mul_keeps_nearby_poses_exact_far_from_the_origin() {
    // B relative to A (Box2D `b2InvMulTransforms`): the positions are subtracted before
    // rotating, so two poses 0.76 m apart stay exact 100 km from the origin, where an f32
    // coordinate is only good to 8 mm.
    let a = Transform2d::from_translation_rotation(Vec2::new(1.0e5, 0.0), Rot2::from_angle(0.35));
    let b = Transform2d::from_translation_rotation(
        Vec2::new(1.0e5 + 0.3125, 0.7),
        Rot2::from_angle(1.1),
    );
    let relative = a.inv_mul(&b);
    let expected = a.rotation.inv_rotate(Vec2::new(0.3125, 0.7));
    assert!(
        (relative.translation - expected).length() < 1e-6,
        "relative position {:?}, expected {expected:?}",
        relative.translation
    );
    assert!((relative.rotation.angle() - 0.75).abs() < 1e-6);

    // Near the origin it is A⁻¹ · B.
    let a = Transform2d::from_translation_rotation(Vec2::new(1.0, 2.0), Rot2::from_angle(0.5));
    let b = Transform2d::from_translation_rotation(Vec2::new(-3.0, 0.5), Rot2::from_angle(-1.0));
    let relative = a.inv_mul(&b);
    let reference = a.inverse().combine(&b);
    assert!((relative.translation - reference.translation).length() < 1e-5);
    assert!((relative.rotation.angle() - reference.rotation.angle()).abs() < 1e-5);
}

#[test]
fn aabb2d_calculates_perimeter_area_containment_and_intersections() {
    let aabb1 = Aabb2d::new(Vec2::new(-1.0, -2.0), Vec2::new(3.0, 4.0));
    assert_eq!(aabb1.perimeter(), 2.0 * (4.0 + 6.0)); // 20.0
    assert_eq!(aabb1.area(), 4.0 * 6.0); // 24.0
    assert_eq!(aabb1.center(), Vec2::new(1.0, 1.0));
    assert_eq!(aabb1.half_extents(), Vec2::new(2.0, 3.0));

    // Containment
    assert!(aabb1.contains_point(Vec2::new(0.0, 0.0)));
    assert!(!aabb1.contains_point(Vec2::new(5.0, 0.0)));

    let inside_box = Aabb2d::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
    assert!(aabb1.contains_aabb(&inside_box));

    // Overlap / intersection
    let overlapping = Aabb2d::new(Vec2::new(2.0, 3.0), Vec2::new(5.0, 6.0));
    assert!(aabb1.intersects(&overlapping));

    let disjoint = Aabb2d::new(Vec2::new(10.0, 10.0), Vec2::new(12.0, 12.0));
    assert!(!aabb1.intersects(&disjoint));

    // Union
    let united = aabb1.union(&overlapping);
    assert_eq!(united.min, Vec2::new(-1.0, -2.0));
    assert_eq!(united.max, Vec2::new(5.0, 6.0));

    // Fatten
    let fat = inside_box.fatten(0.5);
    assert_eq!(fat.min, Vec2::new(-0.5, -0.5));
    assert_eq!(fat.max, Vec2::new(1.5, 1.5));
}

#[test]
fn ray2d_intersects_aabb2d_correctly() {
    let aabb = Aabb2d::new(Vec2::new(2.0, -1.0), Vec2::new(4.0, 1.0));

    // Direct hit
    let ray_hit = Ray2d::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
    let t = ray_hit.intersects_aabb(&aabb);
    assert!(t.is_some());
    assert!((t.unwrap() - 2.0).abs() < 1e-5);

    // Ray pointing away
    let ray_away = Ray2d::new(Vec2::new(0.0, 0.0), Vec2::new(-1.0, 0.0));
    assert!(ray_away.intersects_aabb(&aabb).is_none());

    // Ray missing to the top
    let ray_miss = Ray2d::new(Vec2::new(0.0, 5.0), Vec2::new(1.0, 0.0));
    assert!(ray_miss.intersects_aabb(&aabb).is_none());

    // Ray originating inside AABB
    let ray_inside = Ray2d::new(Vec2::new(3.0, 0.0), Vec2::new(1.0, 0.0));
    let t_inside = ray_inside.intersects_aabb(&aabb);
    assert!(t_inside.is_some());
    assert_eq!(t_inside.unwrap(), 0.0);
}

#[test]
fn rot2_relative_angle_is_continuous_across_pi() {
    let a = Rot2::from_angle(3.0);
    let b = Rot2::from_angle(-3.0);
    // The naive difference of wrapped angles is -6; the relative rotation is 2π - 6.
    assert!((a.angle_to(b) - (2.0 * PI - 6.0)).abs() < 1e-5);
    assert!((b.angle_to(a) + (2.0 * PI - 6.0)).abs() < 1e-5);
    assert!((Rot2::from_angle(0.25).angle_to(Rot2::from_angle(1.0)) - 0.75).abs() < 1e-6);
    assert!(Rot2::from_angle(1.0).angle_to(Rot2::from_angle(1.0)).abs() < 1e-6);
}

#[test]
fn unwind_angle_wraps_into_minus_pi_to_pi() {
    assert!((unwind_angle(0.5) - 0.5).abs() < 1e-6);
    assert!((unwind_angle(-6.0) - (2.0 * PI - 6.0)).abs() < 1e-6);
    assert!((unwind_angle(6.0) - (6.0 - 2.0 * PI)).abs() < 1e-6);
    assert!((unwind_angle(7.0 * PI + 0.25) - (0.25 - PI)).abs() < 1e-5);
    for k in -50..=50 {
        let angle = k as f32 * 0.37;
        let wrapped = unwind_angle(angle);
        assert!((-PI..=PI).contains(&wrapped), "{angle} -> {wrapped}");
        assert!(
            Rot2::from_angle(angle)
                .angle_to(Rot2::from_angle(wrapped))
                .abs()
                < 1e-4
        );
    }
}

#[test]
fn rot2_integrates_small_turns_without_trigonometry() {
    let mut q = Rot2::from_angle(0.3);
    for _ in 0..240 {
        q = q.integrate(2.0 / 240.0);
    }
    // 2 rad in 240 steps of 1/120 rad, each short by about (1/120)^3 / 3.
    assert!((q.angle() - 2.3).abs() < 1e-4, "{}", q.angle());
    assert!(((q.c * q.c + q.s * q.s) - 1.0).abs() < 1e-6);
    assert_eq!(Rot2::IDENTITY.integrate(0.0), Rot2::IDENTITY);
}
