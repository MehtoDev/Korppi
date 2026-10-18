//! Integration tests for 2D Dynamic AABB Tree BVH.

use korppi_math::{Aabb2d, Ray2d, Vec2};
use korppi_physics2d::bvh::DynamicTree2d;

#[test]
fn dynamic_tree_2d_handles_inserts_raycasts_and_queries() {
    let mut tree = DynamicTree2d::new();
    assert_eq!(tree.proxy_count(), 0);

    // Insert 3 boxes
    // Box 1: [0, 0] to [2, 2]
    let id1 = tree.create_proxy(Aabb2d::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0)), 101);
    // Box 2: [5, 5] to [7, 7]
    let id2 = tree.create_proxy(Aabb2d::new(Vec2::new(5.0, 5.0), Vec2::new(7.0, 7.0)), 102);
    // Box 3: [1, 1] to [3, 3] (overlaps Box 1)
    let id3 = tree.create_proxy(Aabb2d::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0)), 103);

    assert_eq!(tree.proxy_count(), 3);

    // Query box overlapping [0.5, 0.5] to [1.5, 1.5] -> should find id1 and id3, but not id2
    let query_box = Aabb2d::new(Vec2::new(0.5, 0.5), Vec2::new(1.5, 1.5));
    let mut hits = Vec::new();
    tree.query(&query_box, |proxy_id, user_data| {
        hits.push((proxy_id, user_data));
        true
    });

    assert_eq!(hits.len(), 2);
    assert!(hits.iter().any(|&(_, u)| u == 101));
    assert!(hits.iter().any(|&(_, u)| u == 103));
    assert!(!hits.iter().any(|&(_, u)| u == 102));

    // Raycast hitting only Box 1 and Box 3 from (-2, 1) towards +X
    let ray = Ray2d::new(Vec2::new(-2.0, 1.0), Vec2::new(1.0, 0.0));
    let mut ray_hits = Vec::new();
    tree.ray_cast(&ray, 10.0, |proxy_id, user_data, hit_fraction| {
        ray_hits.push((proxy_id, user_data, hit_fraction));
        hit_fraction // clip to closest hit
    });

    assert!(!ray_hits.is_empty());
    assert!(ray_hits.iter().any(|&(_, u, _)| u == 101));

    // Move proxy id1 slightly (inside fat AABB margin)
    let moved_slightly = tree.move_proxy(
        id1,
        Aabb2d::new(Vec2::new(0.05, 0.05), Vec2::new(2.05, 2.05)),
        Vec2::ZERO,
    );
    assert!(
        !moved_slightly,
        "Small movement should stay within fat AABB"
    );

    // Move proxy id1 far away
    let moved_far = tree.move_proxy(
        id1,
        Aabb2d::new(Vec2::new(20.0, 20.0), Vec2::new(22.0, 22.0)),
        Vec2::ZERO,
    );
    assert!(moved_far, "Large movement should trigger re-insertion");

    // Query original location should no longer find id1
    let mut hits_after_move = Vec::new();
    tree.query(
        &Aabb2d::new(Vec2::new(0.0, 0.0), Vec2::new(0.5, 0.5)),
        |_, u| {
            hits_after_move.push(u);
            true
        },
    );
    assert!(!hits_after_move.contains(&101));

    // Remove proxies
    tree.destroy_proxy(id1);
    tree.destroy_proxy(id2);
    tree.destroy_proxy(id3);
    assert_eq!(tree.proxy_count(), 0);
}

#[test]
fn dynamic_tree_2d_finds_all_overlapping_pairs_deterministically() {
    let mut tree = DynamicTree2d::new();

    // Box A: [-1, -1] to [1, 1]
    let a = tree.create_proxy(Aabb2d::new(Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0)), 1);
    // Box B: [0.5, 0.5] to [2, 2] (overlaps A)
    let b = tree.create_proxy(Aabb2d::new(Vec2::new(0.5, 0.5), Vec2::new(2.0, 2.0)), 2);
    // Box C: [10, 10] to [12, 12] (disjoint)
    let _c = tree.create_proxy(Aabb2d::new(Vec2::new(10.0, 10.0), Vec2::new(12.0, 12.0)), 3);

    let mut pairs = Vec::new();
    tree.overlapping_pairs(|a, b| pairs.push((a, b)));
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], (a.min(b), a.max(b)));
}
