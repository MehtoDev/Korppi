//! Dynamic 2D shadow atlas shelf packing and two-tier static/dynamic shadow caching (`RND-11`, `RND-13`).

use korppi_math::{Vec2, Vec3};
use std::cell::RefCell;

/// Allocated sub-region (tile) within the 2D shadow atlas texture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasSlot {
    /// Normalized UV minimum corner $[0, 1]^2$.
    pub uv_min: Vec2,
    /// Normalized UV maximum corner $[0, 1]^2$.
    pub uv_max: Vec2,
    /// Pixel rectangle `[offset_x, offset_y, width, height]` within the atlas.
    pub pixel_rect: [u32; 4],
}

impl AtlasSlot {
    /// Remaps local tile-space UV coordinates $[0, 1]^2$ into global atlas UV coordinates.
    #[inline]
    pub fn remap_uv(&self, local_uv: Vec2) -> Vec2 {
        self.uv_min + local_uv * (self.uv_max - self.uv_min)
    }
}

/// Internal shelf record for 2D shelf packing.
#[derive(Clone, Copy, Debug)]
struct Shelf {
    y: u32,
    height: u32,
    current_x: u32,
}

/// Dynamic 2D texture atlas allocator for directional cascades, spotlights, and point light shadows.
pub struct ShadowAtlas {
    size: u32,
    shelves: Vec<Shelf>,
    free_slots: Vec<AtlasSlot>,
}

impl ShadowAtlas {
    /// Creates a new square shadow atlas allocator with the given resolution in texels.
    pub fn new(size: u32) -> Self {
        Self {
            size,
            shelves: Vec::with_capacity(16),
            free_slots: Vec::with_capacity(32),
        }
    }

    /// Total width/height of the square shadow atlas in pixels.
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    #[inline]
    fn make_slot(&self, x: u32, y: u32, tile_resolution: u32) -> AtlasSlot {
        let size_f = self.size as f32;
        let uv_min = Vec2::new(x as f32 / size_f, y as f32 / size_f);
        let uv_max = Vec2::new(
            (x + tile_resolution) as f32 / size_f,
            (y + tile_resolution) as f32 / size_f,
        );
        AtlasSlot {
            uv_min,
            uv_max,
            pixel_rect: [x, y, tile_resolution, tile_resolution],
        }
    }

    /// Allocates a square shadow tile of the given resolution within the atlas.
    pub fn allocate(&mut self, tile_resolution: u32) -> Option<AtlasSlot> {
        if tile_resolution == 0 || tile_resolution > self.size {
            return None;
        }

        // 0. Reuse a previously freed slot of matching resolution if available (`RND-13`)
        if let Some(pos) = self
            .free_slots
            .iter()
            .position(|s| s.pixel_rect[2] == tile_resolution && s.pixel_rect[3] == tile_resolution)
        {
            return Some(self.free_slots.swap_remove(pos));
        }

        // 1. Try to fit into an existing shelf with matching or larger height
        let mut best_shelf_idx: Option<usize> = None;
        let mut min_height_waste = u32::MAX;

        for (idx, shelf) in self.shelves.iter().enumerate() {
            if shelf.height >= tile_resolution && shelf.current_x + tile_resolution <= self.size {
                let waste = shelf.height - tile_resolution;
                if waste < min_height_waste {
                    min_height_waste = waste;
                    best_shelf_idx = Some(idx);
                }
            }
        }

        if let Some(idx) = best_shelf_idx {
            let shelf = &mut self.shelves[idx];
            let x = shelf.current_x;
            let y = shelf.y;
            shelf.current_x += tile_resolution;
            return Some(self.make_slot(x, y, tile_resolution));
        }

        // 2. Open a new shelf at the bottom if vertical space remains
        let next_y = self
            .shelves
            .iter()
            .map(|s| s.y + s.height)
            .max()
            .unwrap_or(0);

        if next_y + tile_resolution <= self.size {
            self.shelves.push(Shelf {
                y: next_y,
                height: tile_resolution,
                current_x: tile_resolution,
            });
            return Some(self.make_slot(0, next_y, tile_resolution));
        }

        // 3. Quadtree-subdivide the smallest larger free slot that is a power-of-two multiple (`RND-24`)
        let mut best_free_idx: Option<usize> = None;
        let mut best_free_size = u32::MAX;
        for (idx, slot) in self.free_slots.iter().enumerate() {
            let sz = slot.pixel_rect[2];
            if sz > tile_resolution
                && slot.pixel_rect[3] == sz
                && sz % tile_resolution == 0
                && (sz / tile_resolution).is_power_of_two()
                && sz < best_free_size
            {
                best_free_size = sz;
                best_free_idx = Some(idx);
            }
        }

        if let Some(idx) = best_free_idx {
            let parent = self.free_slots.swap_remove(idx);
            let base_x = parent.pixel_rect[0];
            let base_y = parent.pixel_rect[1];
            let mut cur_size = parent.pixel_rect[2];
            while cur_size > tile_resolution {
                let half = cur_size / 2;
                let q1 = self.make_slot(base_x + half, base_y, half);
                let q2 = self.make_slot(base_x, base_y + half, half);
                let q3 = self.make_slot(base_x + half, base_y + half, half);
                self.free_slots.push(q3);
                self.free_slots.push(q2);
                self.free_slots.push(q1);
                cur_size = half;
            }
            return Some(self.make_slot(base_x, base_y, tile_resolution));
        }

        None
    }

    /// Returns a previously allocated tile slot to the free list so future lights can reuse it (`RND-13`, `RND-24`).
    ///
    /// Ignores duplicate deallocations of a slot that is already present in `free_slots`.
    #[inline]
    pub fn deallocate(&mut self, slot: AtlasSlot) {
        if !self
            .free_slots
            .iter()
            .any(|s| s.pixel_rect == slot.pixel_rect)
        {
            self.free_slots.push(slot);
        }
    }

    /// Resets all allocations in the atlas for reuse.
    pub fn clear(&mut self) {
        self.shelves.clear();
        self.free_slots.clear();
    }
}

/// Classification of shadow casters for two-tier static caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CasterType {
    /// Static geometry that is cached across frames into the static shadow atlas.
    Static,
    /// Dynamic moving entities that are re-rasterized per frame into the dynamic atlas.
    Dynamic,
}

/// Bounding volume and classification for a shadow caster.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowCaster {
    /// Unique caster identifier.
    pub id: u64,
    /// Static or dynamic classification.
    pub caster_type: CasterType,
    /// Axis-Aligned Bounding Box minimum corner in world/view space.
    pub aabb_min: Vec3,
    /// Axis-Aligned Bounding Box maximum corner in world/view space.
    pub aabb_max: Vec3,
}

/// Blit / copy descriptor copying a cached static shadow tile into the dynamic active frame atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileCopyCommand {
    /// Source rectangle in the Static Cache Atlas `[x, y, width, height]`.
    pub src_rect: [u32; 4],
    /// Destination rectangle in the Dynamic Frame Atlas `[x, y, width, height]`.
    pub dst_rect: [u32; 4],
}

/// State tracking for an individual light's cached shadow tile.
#[derive(Clone, Debug)]
pub struct CachedTile {
    /// Unique light identifier.
    pub light_id: u64,
    /// Resolution of the shadow tile in pixels.
    pub resolution: u32,
    /// Allocated slot in the persistent Static Cache Atlas.
    pub static_slot: AtlasSlot,
    /// Allocated slot in the active Dynamic Frame Atlas.
    pub dynamic_slot: AtlasSlot,
    /// Flag indicating if the static shadow map is dirty and requires re-rasterization.
    pub is_dirty: bool,
    /// IDs of static casters captured in the current static cache.
    pub cached_static_casters: Vec<u64>,
    last_position: Option<Vec3>,
    last_radius: Option<f32>,
    cached_static_bounds: Vec<(u64, Vec3, Vec3)>,
    last_active_frame: u64,
}

struct PlanScratchPool {
    copies: Option<Vec<TileCopyCommand>>,
    static_lights: Option<Vec<(u64, AtlasSlot, Vec<u64>)>>,
    dynamic_lights: Option<Vec<(u64, AtlasSlot, Vec<u64>)>>,
    caster_vecs: Vec<Vec<u64>>,
    bounds_vecs: Vec<Vec<(u64, Vec3, Vec3)>>,
}

impl Default for PlanScratchPool {
    fn default() -> Self {
        Self {
            copies: None,
            static_lights: None,
            dynamic_lights: None,
            caster_vecs: Vec::with_capacity(128),
            bounds_vecs: Vec::with_capacity(128),
        }
    }
}

thread_local! {
    static PLAN_SCRATCH: RefCell<PlanScratchPool> = RefCell::new(PlanScratchPool::default());
}

fn take_caster_vec(min_cap: usize) -> Vec<u64> {
    let mut v = PLAN_SCRATCH
        .with(|cell| cell.borrow_mut().caster_vecs.pop())
        .unwrap_or_default();
    v.clear();
    let target = min_cap.max(64);
    if v.capacity() < target {
        v.reserve(target - v.capacity());
    }
    v
}

fn recycle_caster_vec(mut v: Vec<u64>) {
    if v.capacity() > 0 {
        v.clear();
        PLAN_SCRATCH.with(|cell| {
            let mut pool = cell.borrow_mut();
            if pool.caster_vecs.len() < 128 {
                pool.caster_vecs.push(v);
            }
        });
    }
}

fn take_bounds_vec(min_cap: usize) -> Vec<(u64, Vec3, Vec3)> {
    let mut v = PLAN_SCRATCH
        .with(|cell| cell.borrow_mut().bounds_vecs.pop())
        .unwrap_or_default();
    v.clear();
    let target = min_cap.max(64);
    if v.capacity() < target {
        v.reserve(target - v.capacity());
    }
    v
}

fn recycle_bounds_vec(mut v: Vec<(u64, Vec3, Vec3)>) {
    if v.capacity() > 0 {
        v.clear();
        PLAN_SCRATCH.with(|cell| {
            let mut pool = cell.borrow_mut();
            if pool.bounds_vecs.len() < 128 {
                pool.bounds_vecs.push(v);
            }
        });
    }
}

/// Execution plan generated for shadow rendering in a frame.
#[derive(Clone, Debug, Default)]
pub struct FrameShadowPlan {
    /// GPU blit/copy commands copying clean static tiles into the dynamic frame atlas.
    pub static_tile_copies: Vec<TileCopyCommand>,
    /// Lights requiring static caster rasterization pass into the static cache atlas.
    pub static_raster_lights: Vec<(u64, AtlasSlot, Vec<u64>)>,
    /// Lights requiring dynamic caster rasterization pass into the dynamic frame atlas.
    pub dynamic_raster_lights: Vec<(u64, AtlasSlot, Vec<u64>)>,
    /// Total static draw calls issued this frame.
    pub static_draw_calls: usize,
    /// Total dynamic draw calls issued this frame.
    pub dynamic_draw_calls: usize,
}

impl FrameShadowPlan {
    fn from_thread_local() -> Self {
        PLAN_SCRATCH.with(|cell| {
            let mut pool = cell.borrow_mut();
            let mut static_tile_copies = pool.copies.take().unwrap_or_default();
            static_tile_copies.clear();
            let mut static_raster_lights = pool.static_lights.take().unwrap_or_default();
            static_raster_lights.clear();
            let mut dynamic_raster_lights = pool.dynamic_lights.take().unwrap_or_default();
            dynamic_raster_lights.clear();
            Self {
                static_tile_copies,
                static_raster_lights,
                dynamic_raster_lights,
                static_draw_calls: 0,
                dynamic_draw_calls: 0,
            }
        })
    }

    /// Clears the plan for reuse while recycling any inner caster `Vec<u64>` allocations.
    pub fn clear(&mut self) {
        self.static_tile_copies.clear();
        for (_, _, v) in self.static_raster_lights.drain(..) {
            recycle_caster_vec(v);
        }
        for (_, _, v) in self.dynamic_raster_lights.drain(..) {
            recycle_caster_vec(v);
        }
        self.static_draw_calls = 0;
        self.dynamic_draw_calls = 0;
    }
}

impl Drop for FrameShadowPlan {
    fn drop(&mut self) {
        for (_, _, v) in self.static_raster_lights.drain(..) {
            recycle_caster_vec(v);
        }
        for (_, _, v) in self.dynamic_raster_lights.drain(..) {
            recycle_caster_vec(v);
        }
        let copies = std::mem::take(&mut self.static_tile_copies);
        let static_lights = std::mem::take(&mut self.static_raster_lights);
        let dynamic_lights = std::mem::take(&mut self.dynamic_raster_lights);
        PLAN_SCRATCH.with(|cell| {
            let mut pool = cell.borrow_mut();
            if copies.capacity() > 0 && pool.copies.is_none() {
                pool.copies = Some(copies);
            }
            if static_lights.capacity() > 0 && pool.static_lights.is_none() {
                pool.static_lights = Some(static_lights);
            }
            if dynamic_lights.capacity() > 0 && pool.dynamic_lights.is_none() {
                pool.dynamic_lights = Some(dynamic_lights);
            }
        });
    }
}

/// Two-tier shadow atlas allocator providing static shadow caching and dynamic caster invalidation.
///
/// Maintains a pre-allocated sorted `(u64, CachedTile)` vector for deterministic tile iteration
/// order (`RND-13`) and recycles freed tiles and scratch buffers so steady-state frame preparation
/// and LRU light eviction perform zero heap allocations (`RND-11`, `RND-24`, `RND-26`).
pub struct CachedShadowAtlas {
    size: u32,
    frame_counter: u64,
    static_atlas: ShadowAtlas,
    dynamic_atlas: ShadowAtlas,
    cached_tiles: Vec<(u64, CachedTile)>,
    scratch_static_ids: Vec<u64>,
    scratch_static_bounds: Vec<(u64, Vec3, Vec3)>,
    scratch_dynamic_ids: Vec<u64>,
}

impl Drop for CachedShadowAtlas {
    fn drop(&mut self) {
        for (_, tile) in self.cached_tiles.drain(..) {
            recycle_caster_vec(tile.cached_static_casters);
            recycle_bounds_vec(tile.cached_static_bounds);
        }
    }
}

impl CachedShadowAtlas {
    /// Creates a new two-tier cached shadow atlas with the given atlas resolution.
    pub fn new(size: u32) -> Self {
        Self {
            size,
            frame_counter: 0,
            static_atlas: ShadowAtlas::new(size),
            dynamic_atlas: ShadowAtlas::new(size),
            cached_tiles: Vec::with_capacity(64),
            scratch_static_ids: Vec::with_capacity(64),
            scratch_static_bounds: Vec::with_capacity(64),
            scratch_dynamic_ids: Vec::with_capacity(64),
        }
    }

    /// Selects a power-of-two shadow tile resolution from a light's world radius (`RND-24`).
    #[inline]
    pub fn select_tile_resolution(light_radius: f32, max_atlas_size: u32) -> u32 {
        let base = if light_radius <= 2.5 {
            256
        } else if light_radius <= 25.0 {
            512
        } else {
            1024
        };
        base.min(max_atlas_size.max(64))
    }

    /// Total atlas dimension in pixels.
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Retrieves the cached tile state for a light, if allocated.
    #[inline]
    pub fn get_tile(&self, light_id: u64) -> Option<&CachedTile> {
        self.cached_tiles
            .binary_search_by_key(&light_id, |&(id, _)| id)
            .ok()
            .map(|idx| &self.cached_tiles[idx].1)
    }

    #[inline]
    fn get_tile_mut(&mut self, light_id: u64) -> Option<&mut CachedTile> {
        self.cached_tiles
            .binary_search_by_key(&light_id, |&(id, _)| id)
            .ok()
            .map(|idx| &mut self.cached_tiles[idx].1)
    }

    /// Removes a light from the atlas and returns its static and dynamic tiles to the free list (`RND-13`, `RND-26`).
    pub fn remove_light(&mut self, light_id: u64) -> bool {
        if let Ok(idx) = self
            .cached_tiles
            .binary_search_by_key(&light_id, |&(id, _)| id)
        {
            let (_, tile) = self.cached_tiles.remove(idx);
            self.static_atlas.deallocate(tile.static_slot);
            self.dynamic_atlas.deallocate(tile.dynamic_slot);
            recycle_caster_vec(tile.cached_static_casters);
            recycle_bounds_vec(tile.cached_static_bounds);
            true
        } else {
            false
        }
    }

    /// Evicts the least-recently-used cached light that is not active in `current_frame`.
    fn evict_lru_inactive(&mut self, current_frame: u64, resolution: u32) -> bool {
        let mut victim_id: Option<u64> = None;
        let mut oldest_frame = u64::MAX;
        for &(id, ref tile) in &self.cached_tiles {
            if tile.last_active_frame < current_frame
                && tile.resolution >= resolution
                && tile.last_active_frame < oldest_frame
            {
                oldest_frame = tile.last_active_frame;
                victim_id = Some(id);
            }
        }
        if let Some(id) = victim_id {
            self.remove_light(id)
        } else {
            false
        }
    }

    /// Registers a light for shadow caching or updates its allocation if resolution changes.
    pub fn register_or_update_light(
        &mut self,
        light_id: u64,
        resolution: u32,
    ) -> Result<(), &'static str> {
        if let Some(existing) = self.get_tile(light_id) {
            if existing.resolution == resolution {
                return Ok(());
            }
            self.remove_light(light_id);
        }

        let static_slot = match self.static_atlas.allocate(resolution) {
            Some(slot) => slot,
            None => {
                if self.evict_lru_inactive(self.frame_counter, resolution) {
                    self.static_atlas
                        .allocate(resolution)
                        .ok_or("Failed to allocate static cache atlas slot")?
                } else {
                    return Err("Failed to allocate static cache atlas slot");
                }
            }
        };
        let dynamic_slot = match self.dynamic_atlas.allocate(resolution) {
            Some(slot) => slot,
            None => {
                if self.evict_lru_inactive(self.frame_counter, resolution) {
                    if let Some(slot) = self.dynamic_atlas.allocate(resolution) {
                        slot
                    } else {
                        self.static_atlas.deallocate(static_slot);
                        return Err("Failed to allocate dynamic frame atlas slot");
                    }
                } else {
                    self.static_atlas.deallocate(static_slot);
                    return Err("Failed to allocate dynamic frame atlas slot");
                }
            }
        };

        let tile = CachedTile {
            light_id,
            resolution,
            static_slot,
            dynamic_slot,
            is_dirty: true,
            cached_static_casters: take_caster_vec(0),
            last_position: None,
            last_radius: None,
            cached_static_bounds: take_bounds_vec(0),
            last_active_frame: self.frame_counter,
        };
        match self
            .cached_tiles
            .binary_search_by_key(&light_id, |&(id, _)| id)
        {
            Ok(idx) => {
                let (_, old) = std::mem::replace(&mut self.cached_tiles[idx], (light_id, tile));
                recycle_caster_vec(old.cached_static_casters);
                recycle_bounds_vec(old.cached_static_bounds);
            }
            Err(idx) => self.cached_tiles.insert(idx, (light_id, tile)),
        }

        Ok(())
    }

    /// Marks a light's cached shadow tile as dirty, forcing static re-rasterization next frame.
    pub fn invalidate_light(&mut self, light_id: u64) {
        if let Some(tile) = self.get_tile_mut(light_id) {
            tile.is_dirty = true;
        }
    }

    /// Invalidates any light whose static cache contains the modified static caster or overlaps its bounding box (`RND-13`).
    pub fn invalidate_caster(
        &mut self,
        caster_id: u64,
        caster_aabb_min: Vec3,
        caster_aabb_max: Vec3,
    ) {
        for &mut (_, ref mut tile) in &mut self.cached_tiles {
            if tile.cached_static_casters.contains(&caster_id) {
                tile.is_dirty = true;
                continue;
            }
            if let (Some(pos), Some(radius)) = (tile.last_position, tile.last_radius) {
                if test_sphere_intersects_aabb(pos, radius, caster_aabb_min, caster_aabb_max) {
                    tile.is_dirty = true;
                }
            }
        }
    }

    /// Prepares shadow rendering for the current frame into a caller-provided `FrameShadowPlan`
    /// with zero steady-state heap allocations (`RND-11`, `RND-13`, `RND-24`, `RND-26`).
    pub fn prepare_frame_into(
        &mut self,
        active_lights: &[(u64, Vec3, f32)],
        casters: &[ShadowCaster],
        plan: &mut FrameShadowPlan,
    ) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let current_frame = self.frame_counter;

        plan.clear();
        if plan.static_tile_copies.capacity() < active_lights.len() {
            plan.static_tile_copies
                .reserve(active_lights.len() - plan.static_tile_copies.capacity());
        }
        if plan.static_raster_lights.capacity() < active_lights.len() {
            plan.static_raster_lights
                .reserve(active_lights.len() - plan.static_raster_lights.capacity());
        }
        if plan.dynamic_raster_lights.capacity() < active_lights.len() {
            plan.dynamic_raster_lights
                .reserve(active_lights.len() - plan.dynamic_raster_lights.capacity());
        }

        // Mark all currently active lights that are already cached with `current_frame`
        // so LRU eviction never evicts a light that is active in the current frame.
        for &(light_id, _, _) in active_lights {
            if let Some(tile) = self.get_tile_mut(light_id) {
                tile.last_active_frame = current_frame;
            }
        }

        // Process all active lights in caller order
        for &(light_id, light_pos, light_radius) in active_lights {
            if self.get_tile(light_id).is_none() {
                let res = Self::select_tile_resolution(light_radius, self.size);
                let _ = self.register_or_update_light(light_id, res);
            }

            self.scratch_static_ids.clear();
            self.scratch_static_bounds.clear();
            self.scratch_dynamic_ids.clear();

            for c in casters {
                if test_sphere_intersects_aabb(light_pos, light_radius, c.aabb_min, c.aabb_max) {
                    match c.caster_type {
                        CasterType::Static => {
                            self.scratch_static_ids.push(c.id);
                            self.scratch_static_bounds
                                .push((c.id, c.aabb_min, c.aabb_max));
                        }
                        CasterType::Dynamic => {
                            self.scratch_dynamic_ids.push(c.id);
                        }
                    }
                }
            }

            let tile_idx = match self
                .cached_tiles
                .binary_search_by_key(&light_id, |&(id, _)| id)
            {
                Ok(idx) => idx,
                Err(_) => continue,
            };
            let tile = &mut self.cached_tiles[tile_idx].1;
            tile.last_active_frame = current_frame;

            // Detect light movement/radius change or static caster set/AABB change (`RND-13`)
            let light_moved =
                tile.last_position != Some(light_pos) || tile.last_radius != Some(light_radius);
            let static_changed =
                tile.cached_static_bounds.as_slice() != self.scratch_static_bounds.as_slice();

            if light_moved || static_changed {
                tile.is_dirty = true;
            }

            if tile.is_dirty {
                plan.static_draw_calls += self.scratch_static_ids.len();
                let mut raster_ids = take_caster_vec(self.scratch_static_ids.len());
                raster_ids.extend_from_slice(&self.scratch_static_ids);
                plan.static_raster_lights
                    .push((light_id, tile.static_slot, raster_ids));

                tile.cached_static_casters.clear();
                tile.cached_static_casters
                    .extend_from_slice(&self.scratch_static_ids);
                tile.cached_static_bounds.clear();
                tile.cached_static_bounds
                    .extend_from_slice(&self.scratch_static_bounds);
                tile.last_position = Some(light_pos);
                tile.last_radius = Some(light_radius);
                tile.is_dirty = false;
            }

            // Blit static depth from Static Cache Atlas into Dynamic Frame Atlas
            plan.static_tile_copies.push(TileCopyCommand {
                src_rect: tile.static_slot.pixel_rect,
                dst_rect: tile.dynamic_slot.pixel_rect,
            });

            // Rasterize dynamic casters into Dynamic Frame Atlas
            if !self.scratch_dynamic_ids.is_empty() {
                plan.dynamic_draw_calls += self.scratch_dynamic_ids.len();
                let mut dyn_ids = take_caster_vec(self.scratch_dynamic_ids.len());
                dyn_ids.extend_from_slice(&self.scratch_dynamic_ids);
                plan.dynamic_raster_lights
                    .push((light_id, tile.dynamic_slot, dyn_ids));
            }
        }
    }

    /// Prepares shadow rendering for the current frame, partitioning static and dynamic passes.
    pub fn prepare_frame(
        &mut self,
        active_lights: &[(u64, Vec3, f32)],
        casters: &[ShadowCaster],
    ) -> FrameShadowPlan {
        let mut plan = FrameShadowPlan::from_thread_local();
        self.prepare_frame_into(active_lights, casters, &mut plan);
        plan
    }
}

/// Helper testing intersection between a sphere and an AABB.
#[inline]
fn test_sphere_intersects_aabb(center: Vec3, radius: f32, b_min: Vec3, b_max: Vec3) -> bool {
    let mut dist_sq = 0.0f32;
    for i in 0..3 {
        let c = center[i];
        let min_v = b_min[i];
        let max_v = b_max[i];
        if c < min_v {
            let d = min_v - c;
            dist_sq += d * d;
        } else if c > max_v {
            let d = c - max_v;
            dist_sq += d * d;
        }
    }
    dist_sq <= radius * radius
}
