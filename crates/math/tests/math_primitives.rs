//! Integration tests for korppi_math primitives, transforms, and bounding volumes.

use korppi_math::{Aabb, Mat3, Mat4, Quat, Ray, Sphere, Transform, Vec2, Vec3, Vec4};

#[test]
fn vector_operations_satisfy_euclidean_geometry() {
    let v2 = Vec2::new(3.0, 4.0);
    assert_eq!(v2.length(), 5.0);

    let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v4.dot(v4), 30.0);

    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2_3 = Vec3::new(4.0, 5.0, 6.0);

    // Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(v1.dot(v2_3), 32.0);

    // Cross product: (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
    let cross = v1.cross(v2_3);
    assert_eq!(cross, Vec3::new(-3.0, 6.0, -3.0));
    assert_eq!(cross.dot(v1), 0.0); // Orthogonal
    assert_eq!(cross.dot(v2_3), 0.0); // Orthogonal

    // Normalization
    let unit_x = Vec3::X;
    assert_eq!(unit_x.length(), 1.0);
}

#[test]
fn matrix_operations_satisfy_linear_algebra() {
    let m3 = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0));
    assert_eq!(m3.determinant(), 24.0);
    let inv_m3 = m3.inverse();
    let ident = m3 * inv_m3;
    assert!((ident.x_axis - Vec3::X).length() < 1e-5);

    let m4 = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
    let pt = m4.transform_point3(Vec3::ZERO);
    assert_eq!(pt, Vec3::new(10.0, 20.0, 30.0));
}

#[test]
fn quaternion_rotates_vectors_accurately() {
    let axis = Vec3::Y;
    let angle = std::f32::consts::FRAC_PI_2; // 90 degrees around Y
    let rot = Quat::from_axis_angle(axis, angle);

    let point = Vec3::new(1.0, 0.0, 0.0);
    let rotated = rot * point;

    // 90 deg around Y should rotate +X into -Z in right-handed coordinates
    assert!((rotated - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5);
}

#[test]
fn transform_computes_model_matrix_and_propagates_hierarchies() {
    let parent = Transform {
        translation: Vec3::new(10.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(2.0),
    };

    let child = Transform {
        translation: Vec3::new(1.0, 2.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    let combined = parent.combine(&child);
    // Scaled child translation by 2 -> (2, 4, 0), then add parent (10, 0, 0) -> (12, 4, 0)
    assert_eq!(combined.translation, Vec3::new(12.0, 4.0, 0.0));
    assert_eq!(combined.scale, Vec3::splat(2.0));

    // Matrix transformation of point
    let mat = combined.to_matrix();
    let local_pt = Vec3::new(0.0, 0.0, 0.0);
    let world_pt = mat.transform_point3(local_pt);
    assert_eq!(world_pt, Vec3::new(12.0, 4.0, 0.0));
}

#[test]
fn aabb_calculates_bounds_surface_area_and_overlap() {
    let aabb1 = Aabb::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(aabb1.center(), Vec3::ZERO);
    assert_eq!(aabb1.extents(), Vec3::splat(2.0));
    assert_eq!(aabb1.half_extents(), Vec3::splat(1.0));
    // Surface area of 2x2x2 cube = 6 * (2 * 2) = 24
    assert_eq!(aabb1.surface_area(), 24.0);

    let aabb2 = Aabb::from_min_max(Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.5, 2.5, 2.5));
    assert!(aabb1.intersects(&aabb2));

    let disjoint_aabb = Aabb::from_min_max(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
    assert!(!aabb1.intersects(&disjoint_aabb));

    // Merge union
    let merged = aabb1.union(&aabb2);
    assert_eq!(merged.min, Vec3::new(-1.0, -1.0, -1.0));
    assert_eq!(merged.max, Vec3::new(2.5, 2.5, 2.5));
}

#[test]
fn sphere_intersects_spheres_and_aabbs() {
    let s1 = Sphere::new(Vec3::ZERO, 1.0);
    let s2 = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert!(s1.intersects_sphere(&s2));

    let s3 = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(!s1.intersects_sphere(&s3));

    let aabb = Aabb::from_min_max(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
    assert!(s1.intersects_aabb(&aabb));
}

#[test]
fn ray_intersects_aabb_correctly() {
    let aabb = Aabb::from_min_max(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, 3.0));
    let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::Z);

    let hit = aabb.intersect_ray(&ray, 0.0, 10.0);
    assert!(hit.is_some());
    let t = hit.unwrap();
    assert!((t - 1.0).abs() < 1e-5); // Hits near face at z = 1.0

    // Missing ray
    let miss_ray = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z);
    assert!(aabb.intersect_ray(&miss_ray, 0.0, 10.0).is_none());
}
