//! Collider attachment defining shape geometry and physical surface properties.

use crate::joint::Frame3d;
use crate::shape::Shape;
use korppi_ecs::Component;
use korppi_math::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Which colliders collide with each other (Box3D `b3Filter`). Two colliders in the same
/// non-zero group always collide if the group is positive and never if it is negative;
/// otherwise each one's `mask_bits` must include the other's `category_bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Filter3d {
    /// The categories the collider belongs to, usually one bit.
    pub category_bits: u64,
    /// The categories the collider collides with.
    pub mask_bits: u64,
    /// Colliders sharing a non-zero group always (positive) or never (negative) collide,
    /// whatever their bits; ragdolls use a negative group to keep their parts apart.
    pub group_index: i32,
}

impl Default for Filter3d {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Filter3d {
    /// Every category, colliding with everything, no group (Box3D `b3DefaultFilter`).
    pub const DEFAULT: Self = Self {
        category_bits: u64::MAX,
        mask_bits: u64::MAX,
        group_index: 0,
    };

    /// A filter with these bits and no group.
    #[must_use]
    pub const fn new(category_bits: u64, mask_bits: u64) -> Self {
        Self {
            category_bits,
            mask_bits,
            group_index: 0,
        }
    }

    /// Sets the group index in builder style.
    #[must_use]
    pub const fn with_group(mut self, group_index: i32) -> Self {
        self.group_index = group_index;
        self
    }

    /// Whether colliders with these filters collide (Box3D `b3ShouldShapesCollide`).
    #[must_use]
    pub const fn should_collide(&self, other: &Self) -> bool {
        if self.group_index == other.group_index && self.group_index != 0 {
            return self.group_index > 0;
        }
        (self.mask_bits & other.category_bits) != 0 && (self.category_bits & other.mask_bits) != 0
    }

    /// Whether a query with `query` sees a collider with this filter (Box3D
    /// `b3ShouldQueryCollide`): each side's mask must accept the other's category.
    #[must_use]
    pub const fn should_query(&self, query: &QueryFilter3d) -> bool {
        (self.category_bits & query.mask_bits) != 0 && (self.mask_bits & query.category_bits) != 0
    }
}

/// Which colliders a query sees (Box3D `b3QueryFilter`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryFilter3d {
    /// The categories of the query, usually one bit.
    pub category_bits: u64,
    /// The categories of colliders the query accepts.
    pub mask_bits: u64,
}

impl Default for QueryFilter3d {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl QueryFilter3d {
    /// Every category, seeing everything (Box3D `b3DefaultQueryFilter`).
    pub const DEFAULT: Self = Self {
        category_bits: u64::MAX,
        mask_bits: u64::MAX,
    };

    /// A query filter with these bits.
    #[must_use]
    pub const fn new(category_bits: u64, mask_bits: u64) -> Self {
        Self {
            category_bits,
            mask_bits,
        }
    }
}

/// Defines collision geometry, friction, and restitution attached to a rigid body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Collider {
    /// The geometric collision primitive.
    pub shape: Shape,
    /// Coulomb friction coefficient (typically 0.0 to 1.0).
    pub friction: f32,
    /// Restitution (bounciness) coefficient (0.0 for inelastic, 1.0 for perfectly elastic).
    pub restitution: f32,
    /// Rolling resistance, usually 0 to 1 (Box3D `rollingResistance`, default 0). Scaled by the
    /// larger rounded radius of a contact, it resists the bodies' relative rotation, so rolling
    /// balls slow down.
    #[serde(default)]
    pub rolling_resistance: f32,
    /// If `true`, the collider makes no contacts; with `enable_sensor_events` it reports the
    /// colliders that overlap it (Box3D `isSensor`). It still gives its body mass.
    pub is_sensor: bool,
    /// Take part in sensor events (Box3D `enableSensorEvents`, off by default, even for
    /// sensors): a sensor reports a collider only if both enable them.
    #[serde(default)]
    pub enable_sensor_events: bool,
    /// Take part in contact begin and end events (Box3D `enableContactEvents`, off by default).
    #[serde(default)]
    pub enable_contact_events: bool,
    /// Take part in contact hit events (Box3D `enableHitEvents`, off by default).
    #[serde(default)]
    pub enable_hit_events: bool,
    /// Which colliders it collides with, and which queries see it.
    #[serde(default)]
    pub filter: Filter3d,
    /// Density in kg/m³ (Box3D `density`, default 1000, water's): a dynamic body's mass,
    /// centre of mass and inertia are its colliders' sums. Zero adds no mass.
    #[serde(default = "default_density")]
    pub density: f32,
    /// Where the shape sits on its body, in the body's frame (Box3D places shapes by their own
    /// coordinates or a transform).
    #[serde(default)]
    pub local_frame: Frame3d,
    /// Tangent velocity for conveyor belts (Box3D `b3SurfaceMaterial::tangentVelocity`). Local to
    /// the shape and projected onto the contact surface.
    #[serde(default)]
    pub tangent_velocity: Vec3,
    /// User material identifier (Box3D `b3SurfaceMaterial::userMaterialId`).
    #[serde(default)]
    pub user_material_id: u64,
    /// Surface materials for mesh shapes (per triangle) or height fields (Box3D `b3ShapeDef::materials`).
    #[serde(default)]
    pub materials: Vec<crate::material::SurfaceMaterial3d>,
}

/// Box3D's default shape density, water's, in kg/m³.
pub const DEFAULT_DENSITY: f32 = 1000.0;

fn default_density() -> f32 {
    DEFAULT_DENSITY
}

/// A shape's mass, centre and inertia tensor about that centre (Box3D `b3MassData`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassData3d {
    /// Mass in kg.
    pub mass: f32,
    /// Centre of mass.
    pub center: Vec3,
    /// Inertia tensor about the centre of mass, kg m².
    pub inertia: Mat3,
}

impl Component for Collider {}

impl Default for Collider {
    fn default() -> Self {
        Self {
            shape: Shape::Sphere { radius: 0.5 },
            friction: 0.3,
            restitution: 0.0,
            rolling_resistance: 0.0,
            is_sensor: false,
            enable_sensor_events: false,
            enable_contact_events: false,
            enable_hit_events: false,
            filter: Filter3d::DEFAULT,
            density: DEFAULT_DENSITY,
            local_frame: Frame3d::IDENTITY,
            tangent_velocity: Vec3::ZERO,
            user_material_id: 0,
            materials: Vec::new(),
        }
    }
}

impl Collider {
    /// Creates a new collider with the given shape.
    #[must_use]
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            ..Default::default()
        }
    }

    /// Sets the friction coefficient.
    #[must_use]
    pub fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    /// Sets the restitution coefficient.
    #[must_use]
    pub fn with_restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    /// Sets the rolling resistance.
    #[must_use]
    pub fn with_rolling_resistance(mut self, rolling_resistance: f32) -> Self {
        self.rolling_resistance = rolling_resistance;
        self
    }

    /// Sets whether this collider acts as a sensor (trigger).
    #[must_use]
    pub fn with_sensor(mut self, is_sensor: bool) -> Self {
        self.is_sensor = is_sensor;
        self
    }

    /// Sets the collision filter.
    #[must_use]
    pub fn with_filter(mut self, filter: Filter3d) -> Self {
        self.filter = filter;
        self
    }

    /// Sets whether this collider takes part in sensor events, as a sensor or a visitor.
    #[must_use]
    pub fn with_sensor_events(mut self, enable: bool) -> Self {
        self.enable_sensor_events = enable;
        self
    }

    /// Sets whether this collider takes part in contact begin and end events.
    #[must_use]
    pub const fn with_contact_events(mut self, enable: bool) -> Self {
        self.enable_contact_events = enable;
        self
    }

    /// Sets whether this collider takes part in contact hit events.
    #[must_use]
    pub const fn with_hit_events(mut self, enable: bool) -> Self {
        self.enable_hit_events = enable;
        self
    }

    /// Sets the density, in kg/m³.
    #[must_use]
    pub fn with_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    /// Sets the density that makes the shape weigh `mass` kg; a shape without volume keeps its
    /// density.
    #[must_use]
    pub fn with_mass(mut self, mass: f32) -> Self {
        let (unit_mass, _, _) = self.shape.mass_data(1.0);
        if unit_mass > 0.0 {
            self.density = mass / unit_mass;
        }
        self
    }

    /// Places the shape on its body.
    #[must_use]
    pub fn with_local_frame(mut self, frame: Frame3d) -> Self {
        self.local_frame = frame;
        self
    }

    /// Sets the local surface tangent velocity for conveyor belts (Box3D `b3SurfaceMaterial::tangentVelocity`).
    #[must_use]
    pub fn with_tangent_velocity(mut self, tangent_velocity: Vec3) -> Self {
        self.tangent_velocity = tangent_velocity;
        self
    }

    /// Sets the user material identifier (Box3D `b3SurfaceMaterial::userMaterialId`).
    #[must_use]
    pub const fn with_user_material_id(mut self, user_material_id: u64) -> Self {
        self.user_material_id = user_material_id;
        self
    }

    /// Sets the base surface material properties from a [`crate::material::SurfaceMaterial3d`].
    #[must_use]
    pub fn with_surface_material(mut self, mat: crate::material::SurfaceMaterial3d) -> Self {
        self.friction = mat.friction;
        self.restitution = mat.restitution;
        self.rolling_resistance = mat.rolling_resistance;
        self.tangent_velocity = mat.tangent_velocity;
        self.user_material_id = mat.user_material_id;
        self
    }

    /// Sets the list of surface materials for mesh shapes (per triangle) or height fields.
    #[must_use]
    pub fn with_mesh_materials(
        mut self,
        materials: Vec<crate::material::SurfaceMaterial3d>,
    ) -> Self {
        self.materials = materials;
        self
    }

    /// Returns the base surface material (Box3D `b3Shape_GetSurfaceMaterial`).
    #[must_use]
    pub fn surface_material(&self) -> crate::material::SurfaceMaterial3d {
        crate::material::SurfaceMaterial3d {
            friction: self.friction,
            restitution: self.restitution,
            rolling_resistance: self.rolling_resistance,
            tangent_velocity: self.tangent_velocity,
            user_material_id: self.user_material_id,
            custom_color: 0,
        }
    }

    /// Updates the base surface material (Box3D `b3Shape_SetSurfaceMaterial`).
    pub fn set_surface_material(&mut self, mat: crate::material::SurfaceMaterial3d) {
        self.friction = mat.friction;
        self.restitution = mat.restitution;
        self.rolling_resistance = mat.rolling_resistance;
        self.tangent_velocity = mat.tangent_velocity;
        self.user_material_id = mat.user_material_id;
    }

    /// Returns the number of mesh materials (Box3D `b3Shape_GetMeshMaterialCount`).
    #[must_use]
    pub fn mesh_material_count(&self) -> usize {
        self.materials.len()
    }

    /// Returns the mesh material at `index` (Box3D `b3Shape_GetMeshSurfaceMaterial`).
    #[must_use]
    pub fn mesh_material(&self, index: usize) -> Option<crate::material::SurfaceMaterial3d> {
        self.materials.get(index).copied()
    }

    /// Updates the mesh material at `index` (Box3D `b3Shape_SetMeshMaterial`).
    pub fn set_mesh_material(&mut self, index: usize, mat: crate::material::SurfaceMaterial3d) {
        if index < self.materials.len() {
            self.materials[index] = mat;
        }
    }

    /// The shape's mass data in the body's frame (Box3D `b3ComputeShapeMass`, moved by the
    /// local frame).
    #[must_use]
    pub fn mass_data(&self) -> MassData3d {
        let (mass, center, inertia) = self.shape.mass_data(self.density);
        let rotation = Mat3::from_quat(self.local_frame.rotation);
        MassData3d {
            mass,
            center: self.local_frame.position + rotation * center,
            inertia: rotation * inertia * rotation.transpose(),
        }
    }

    /// The shape's pose in the world on a body at `position` turned by `rotation`.
    #[must_use]
    pub fn world_pose(&self, position: Vec3, rotation: Quat) -> (Vec3, Quat) {
        (
            position + rotation * self.local_frame.position,
            rotation * self.local_frame.rotation,
        )
    }

    /// The shape's extents about `origin` in the body's frame (Box3D `b3ComputeShapeExtent`):
    /// the radius of a sphere inside the shape, and the largest reach per axis.
    #[must_use]
    pub fn extents_about(&self, origin: Vec3) -> (f32, Vec3) {
        let frame = &self.local_frame;
        let reach = |p: Vec3| (frame.position + frame.rotation * p - origin).abs();
        let max_reach = |points: &[Vec3]| points.iter().fold(Vec3::ZERO, |m, &p| m.max(reach(p)));
        let (min_extent, _) = self.shape.extents();
        let max_extent = match &self.shape {
            Shape::Sphere { radius } => reach(Vec3::ZERO) + Vec3::splat(*radius),
            Shape::Capsule {
                radius,
                half_height,
            } => {
                let ends = [
                    Vec3::new(0.0, -half_height, 0.0),
                    Vec3::new(0.0, *half_height, 0.0),
                ];
                max_reach(&ends) + Vec3::splat(*radius)
            }
            Shape::Box { half_extents } => {
                let h = *half_extents;
                let corners: [Vec3; 8] = std::array::from_fn(|i| {
                    Vec3::new(
                        if i & 1 == 0 { h.x } else { -h.x },
                        if i & 2 == 0 { h.y } else { -h.y },
                        if i & 4 == 0 { h.z } else { -h.z },
                    )
                });
                max_reach(&corners)
            }
            Shape::ConvexHull(hull) => max_reach(hull.points()),
            Shape::TriangleMesh(mesh) => max_reach(mesh.vertices()),
            Shape::HeightField(hf) => max_reach(&[hf.bounds().0, hf.bounds().1]),
            Shape::Compound(comp) => max_reach(&[comp.bounds().0, comp.bounds().1]),
            Shape::Triangle { a, b, c } => max_reach(&[*a, *b, *c]),
        };
        (min_extent, max_extent)
    }
}
