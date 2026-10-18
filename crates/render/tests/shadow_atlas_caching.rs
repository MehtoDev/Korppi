//! Verification test for Static vs. Dynamic Shadow Atlas Caching (AUDIT-4.3 / Finding 3.3).
//!
//! Research reference: `research/render/shadow_mapping_and_caching_architectures.md:180-210`.
//! Static casters are cached in a persistent static atlas and blitted per frame;
//! only dynamic casters are rasterized per frame, achieving >= 60% draw call reduction.

use korppi_math::Vec3;
use korppi_render::shadows::{CachedShadowAtlas, CasterType, ShadowCaster};

#[test]
fn test_cached_shadow_atlas_allocation_and_tile_slots() {
    let mut cached_atlas = CachedShadowAtlas::new(4096);
    let light_id = 1u64;
    let resolution = 512;

    cached_atlas
        .register_or_update_light(light_id, resolution)
        .expect("Register light failed");

    let tile = cached_atlas.get_tile(light_id).expect("Tile must exist");
    assert_eq!(tile.resolution, resolution);
    assert!(tile.is_dirty, "Newly registered tile must be dirty");
    assert_eq!(tile.static_slot.pixel_rect[2], resolution);
    assert_eq!(tile.dynamic_slot.pixel_rect[2], resolution);
}

#[test]
fn test_static_shadow_caching_draw_call_reduction() {
    let mut cached_atlas = CachedShadowAtlas::new(4096);
    let light_id = 100u64;
    let light_pos = Vec3::new(0.0, 5.0, 0.0);
    let light_radius = 10.0;
    let resolution = 512;

    cached_atlas
        .register_or_update_light(light_id, resolution)
        .expect("Register light failed");

    // Create 10 static casters and 2 dynamic casters inside the light volume
    let mut casters = Vec::new();
    for i in 0..10 {
        casters.push(ShadowCaster {
            id: i as u64,
            caster_type: CasterType::Static,
            aabb_min: Vec3::new(i as f32 - 5.0, 0.0, -1.0),
            aabb_max: Vec3::new(i as f32 - 4.0, 2.0, 1.0),
        });
    }
    for j in 10..12 {
        casters.push(ShadowCaster {
            id: j as u64,
            caster_type: CasterType::Dynamic,
            aabb_min: Vec3::new(j as f32 - 10.0, 0.0, -1.0),
            aabb_max: Vec3::new(j as f32 - 9.0, 1.8, 1.0),
        });
    }

    let active_lights = [(light_id, light_pos, light_radius)];

    // Frame 0: Cold cache / initialization
    let plan_frame_0 = cached_atlas.prepare_frame(&active_lights, &casters);
    assert_eq!(
        plan_frame_0.static_draw_calls, 10,
        "Frame 0 must render all 10 static casters"
    );
    assert_eq!(
        plan_frame_0.dynamic_draw_calls, 2,
        "Frame 0 must render 2 dynamic casters"
    );
    let total_draw_calls_frame_0 = plan_frame_0.static_draw_calls + plan_frame_0.dynamic_draw_calls;
    assert_eq!(total_draw_calls_frame_0, 12);

    // Frame 1: Warm cache (static geometry unchanged)
    let plan_frame_1 = cached_atlas.prepare_frame(&active_lights, &casters);
    assert_eq!(
        plan_frame_1.static_draw_calls, 0,
        "Frame 1 must skip all static casters (using cached static depth)"
    );
    assert_eq!(
        plan_frame_1.dynamic_draw_calls, 2,
        "Frame 1 renders only dynamic casters"
    );
    assert_eq!(
        plan_frame_1.static_tile_copies.len(),
        1,
        "Frame 1 must generate 1 blit/copy command from static cache to dynamic atlas"
    );

    let total_draw_calls_frame_1 = plan_frame_1.static_draw_calls + plan_frame_1.dynamic_draw_calls;
    assert_eq!(total_draw_calls_frame_1, 2);

    // Verify >= 60% draw call reduction
    let reduction = 1.0 - (total_draw_calls_frame_1 as f32 / total_draw_calls_frame_0 as f32);
    assert!(
        reduction >= 0.60,
        "Draw call reduction must be >= 60%, got {:.1}%",
        reduction * 100.0
    );
}

#[test]
fn test_cache_invalidation_when_static_geometry_moves() {
    let mut cached_atlas = CachedShadowAtlas::new(4096);
    let light_id = 42u64;
    let light_pos = Vec3::ZERO;
    let light_radius = 8.0;

    cached_atlas
        .register_or_update_light(light_id, 512)
        .expect("Register failed");

    let casters = vec![
        ShadowCaster {
            id: 1,
            caster_type: CasterType::Static,
            aabb_min: Vec3::new(-1.0, -1.0, -1.0),
            aabb_max: Vec3::new(1.0, 1.0, 1.0),
        },
        ShadowCaster {
            id: 2,
            caster_type: CasterType::Dynamic,
            aabb_min: Vec3::new(2.0, 0.0, 0.0),
            aabb_max: Vec3::new(3.0, 1.0, 1.0),
        },
    ];

    let active_lights = [(light_id, light_pos, light_radius)];

    // Frame 0: Initial render -> cached
    let p0 = cached_atlas.prepare_frame(&active_lights, &casters);
    assert_eq!(p0.static_draw_calls, 1);

    // Frame 1: Cached -> zero static draw calls
    let p1 = cached_atlas.prepare_frame(&active_lights, &casters);
    assert_eq!(p1.static_draw_calls, 0);

    // Static object 1 moves or transforms -> invalidate
    cached_atlas.invalidate_caster(1, Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

    // Frame 2: Must re-render static caster
    let p2 = cached_atlas.prepare_frame(&active_lights, &casters);
    assert_eq!(
        p2.static_draw_calls, 1,
        "Invalidated static cache must re-render static casters"
    );

    // Frame 3: Cached again!
    let p3 = cached_atlas.prepare_frame(&active_lights, &casters);
    assert_eq!(p3.static_draw_calls, 0);
}

#[test]
fn test_dynamic_geometry_movement_does_not_invalidate_static_cache() {
    let mut cached_atlas = CachedShadowAtlas::new(4096);
    let light_id = 77u64;
    let light_pos = Vec3::ZERO;
    let light_radius = 10.0;

    cached_atlas
        .register_or_update_light(light_id, 512)
        .expect("Register failed");

    let mut casters = vec![
        ShadowCaster {
            id: 10,
            caster_type: CasterType::Static,
            aabb_min: Vec3::new(-1.0, -1.0, -1.0),
            aabb_max: Vec3::new(1.0, 1.0, 1.0),
        },
        ShadowCaster {
            id: 20,
            caster_type: CasterType::Dynamic,
            aabb_min: Vec3::new(2.0, 0.0, 0.0),
            aabb_max: Vec3::new(3.0, 1.0, 1.0),
        },
    ];

    let active_lights = [(light_id, light_pos, light_radius)];

    // Frame 0: Warmup
    cached_atlas.prepare_frame(&active_lights, &casters);

    // Frame 1: Moving dynamic caster from X=2.0 to X=5.0
    casters[1].aabb_min = Vec3::new(5.0, 0.0, 0.0);
    casters[1].aabb_max = Vec3::new(6.0, 1.0, 1.0);

    let p1 = cached_atlas.prepare_frame(&active_lights, &casters);
    assert_eq!(
        p1.static_draw_calls, 0,
        "Moving dynamic casters must NOT invalidate static shadow cache"
    );
    assert_eq!(p1.dynamic_draw_calls, 1);
}

#[test]
fn test_g12_static_caster_added_later_and_moving_light_trigger_rerender() {
    let caster = |id: u64, x: f32| ShadowCaster {
        id,
        caster_type: CasterType::Static,
        aabb_min: Vec3::new(x, 0.0, 0.0),
        aabb_max: Vec3::new(x + 1.0, 1.0, 1.0),
    };
    let mut atlas = CachedShadowAtlas::new(4096);

    // Frame 1: initial static caster 10 in range
    let f1 = atlas.prepare_frame(&[(1, Vec3::ZERO, 10.0)], &[caster(10, 2.0)]);
    assert_eq!(f1.static_raster_lights.len(), 1);
    assert_eq!(f1.static_raster_lights[0].2, vec![10]);

    // Frame 2: static caster 11 placed in range -> must re-render static tile with [10, 11]
    let f2 = atlas.prepare_frame(
        &[(1, Vec3::ZERO, 10.0)],
        &[caster(10, 2.0), caster(11, 3.0)],
    );
    assert_eq!(
        f2.static_raster_lights.len(),
        1,
        "G12: adding static caster 11 in range must trigger static shadow re-render"
    );
    assert_eq!(f2.static_raster_lights[0].2, vec![10, 11]);

    // Frame 3: light 1 moves 100 m to x = 100.0 where static caster 12 is -> must re-render [12]
    let f3 = atlas.prepare_frame(
        &[(1, Vec3::new(100.0, 0.0, 0.0), 10.0)],
        &[caster(12, 101.0)],
    );
    assert_eq!(
        f3.static_raster_lights.len(),
        1,
        "G12: moving light 100 m must invalidate its cached static shadow tile"
    );
    assert_eq!(f3.static_raster_lights[0].2, vec![12]);
}

#[test]
fn test_g12_100_sequential_lights_all_receive_atlas_tiles() {
    let mut atlas = CachedShadowAtlas::new(4096);
    for id in 0..100u64 {
        let plan = atlas.prepare_frame(&[(id, Vec3::ZERO, 5.0)], &[]);
        assert!(
            atlas.get_tile(id).is_some(),
            "G12: active light {id} must receive a shadow atlas tile (inactive tiles must be evicted)"
        );
        assert_eq!(
            plan.static_tile_copies.len(),
            1,
            "G12: active light {id} must have 1 static tile copy command"
        );
    }
}

#[test]
fn test_rnd13_invalidate_caster_uses_aabb_overlap_and_layout_is_deterministic() {
    let mut atlas = CachedShadowAtlas::new(4096);
    let lights = [
        (1u64, Vec3::new(0.0, 0.0, 0.0), 5.0),
        (2u64, Vec3::new(50.0, 0.0, 0.0), 5.0),
    ];
    let casters = [ShadowCaster {
        id: 10,
        caster_type: CasterType::Static,
        aabb_min: Vec3::new(1.0, 0.0, 0.0),
        aabb_max: Vec3::new(2.0, 1.0, 1.0),
    }];

    // Frame 0: warm up cache
    let _ = atlas.prepare_frame(&lights, &casters);
    // Frame 1: both lights clean
    let p1 = atlas.prepare_frame(&lights, &casters);
    assert_eq!(p1.static_raster_lights.len(), 0);

    // Invalidate a caster ID (99) not previously in light 1's cached_static_casters,
    // but whose AABB overlaps light 1 (and NOT light 2).
    atlas.invalidate_caster(99, Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0));

    let p2 = atlas.prepare_frame(&lights, &casters);
    assert_eq!(
        p2.static_raster_lights.len(),
        1,
        "RND-13: invalidate_caster must invalidate light 1 via AABB intersection"
    );
    assert_eq!(p2.static_raster_lights[0].0, 1);

    // Deterministic tile placement across two independent atlases
    let mut atlas_a = CachedShadowAtlas::new(4096);
    let mut atlas_b = CachedShadowAtlas::new(4096);
    let many_lights: Vec<(u64, Vec3, f32)> = (0..16)
        .map(|i| (100 - i, Vec3::new(i as f32 * 4.0, 0.0, 0.0), 3.0))
        .collect();
    let _ = atlas_a.prepare_frame(&many_lights, &casters);
    let _ = atlas_b.prepare_frame(&many_lights, &casters);
    for &(id, _, _) in &many_lights {
        assert_eq!(
            atlas_a.get_tile(id).unwrap().static_slot,
            atlas_b.get_tile(id).unwrap().static_slot,
            "RND-13: tile placement must be deterministic across runs"
        );
    }
}
