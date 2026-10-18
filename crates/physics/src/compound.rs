//! Box3D's compound shapes (`b3CompoundData`, `b3CreateCompound`, `compound.c`).
//!
//! A static-only aggregate shape containing multiple child shapes with an internal BVH,
//! multi-manifold contacts across child shapes, and front-to-back ray casting (P3D-13, P3D-10).

use crate::math::modified_cross;
use crate::shape::Shape;
use korppi_math::{Quat, Ray, Vec3};
use serde::{Deserialize, Serialize};

/// A child shape positioned and oriented inside a [`Compound3d`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompoundChild {
    /// Collision shape primitive of the child.
    pub shape: Shape,
    /// Local position relative to the compound origin.
    pub position: Vec3,
    /// Local orientation relative to the compound frame.
    pub rotation: Quat,
}

impl CompoundChild {
    /// Creates a new child with `shape`, `position`, and `rotation`.
    #[must_use]
    pub const fn new(shape: Shape, position: Vec3, rotation: Quat) -> Self {
        Self {
            shape,
            position,
            rotation,
        }
    }

    /// Computes the axis-aligned bounding box of this child in compound space.
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let aabb = self.shape.compute_aabb(self.position, self.rotation);
        (aabb.min, aabb.max)
    }
}

/// Hit result from ray casting against a [`Compound3d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompoundRayHit {
    /// Fraction along the ray translation (0.0 to 1.0).
    pub fraction: f32,
    /// Hit point in compound space.
    pub point: Vec3,
    /// Surface normal in compound space at the hit point.
    pub normal: Vec3,
    /// Index of the child shape hit.
    pub child_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct CompoundNode {
    lower: Vec3,
    upper: Vec3,
    child_index: u32,
    axis: u8,
    left: u32,
    right: u32,
}

impl CompoundNode {
    const fn is_leaf(&self) -> bool {
        self.child_index != u32::MAX
    }
}

/// Box3D `b3TestBoundsOverlap`.
fn bounds_overlap(lower1: Vec3, upper1: Vec3, lower2: Vec3, upper2: Vec3) -> bool {
    (lower2 - upper1)
        .max(lower1 - upper2)
        .cmple(Vec3::ZERO)
        .all()
}

/// Box3D `b3TestBoundsRayOverlap`.
fn bounds_ray_overlap(lower: Vec3, upper: Vec3, start: Vec3, delta: Vec3) -> bool {
    let center = (lower + upper) * 0.5;
    let extent = upper - center;
    let start = start - center;
    let separation = delta.cross(start).abs() - modified_cross(delta.abs(), extent);
    separation.cmple(Vec3::ZERO).all()
}

/// A static aggregate shape containing multiple child shapes with an internal BVH (Box3D `b3CompoundData`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Compound3d {
    children: Vec<CompoundChild>,
    nodes: Vec<CompoundNode>,
    root: u32,
    lower: Vec3,
    upper: Vec3,
}

impl Compound3d {
    /// Builds a compound shape containing `children`.
    ///
    /// Returns `None` if `children` is empty.
    #[must_use]
    pub fn new(children: Vec<CompoundChild>) -> Option<Self> {
        if children.is_empty() {
            return None;
        }

        let mut child_indices: Vec<usize> = (0..children.len()).collect();
        let mut nodes = Vec::with_capacity(2 * children.len());
        let root = build_bvh(&children, &mut child_indices, &mut nodes);

        let root_node = &nodes[root as usize];
        let lower = root_node.lower;
        let upper = root_node.upper;

        Some(Self {
            children,
            nodes,
            root,
            lower,
            upper,
        })
    }

    /// Number of child shapes.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Access child shape by index.
    #[must_use]
    pub fn child(&self, index: usize) -> &CompoundChild {
        &self.children[index]
    }

    /// Slice of all child shapes.
    #[must_use]
    pub fn children(&self) -> &[CompoundChild] {
        &self.children
    }

    /// Local axis-aligned bounding box containing all children.
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.lower, self.upper)
    }

    /// Spatial query finding all children intersecting `[lower, upper]`.
    pub fn query<F: FnMut(usize) -> bool>(&self, lower: Vec3, upper: Vec3, mut callback: F) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = [0u32; 64];
        let mut top = 1;
        stack[0] = self.root;

        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];
            if bounds_overlap(node.lower, node.upper, lower, upper) {
                if node.is_leaf() {
                    if !callback(node.child_index as usize) {
                        return;
                    }
                } else if top + 2 <= stack.len() {
                    stack[top] = node.left;
                    top += 1;
                    stack[top] = node.right;
                    top += 1;
                }
            }
        }
    }

    /// Front-to-back ray casting through the internal BVH.
    #[must_use]
    pub fn ray_cast(
        &self,
        origin: Vec3,
        translation: Vec3,
        max_fraction: f32,
    ) -> Option<CompoundRayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let trans_len = translation.length();
        if trans_len <= 1e-7 {
            return None;
        }
        let ray_dir = translation / trans_len;

        let mut best: Option<CompoundRayHit> = None;
        let mut fraction = max_fraction;

        let bounds = |fraction: f32| {
            let end = origin + translation * fraction;
            (origin.min(end), origin.max(end))
        };
        let (mut ray_lower, mut ray_upper) = bounds(fraction);

        let mut stack = [0u32; 64];
        let mut top = 1;
        stack[0] = self.root;

        while top > 0 {
            top -= 1;
            let node = &self.nodes[stack[top] as usize];

            if bounds_overlap(node.lower, node.upper, ray_lower, ray_upper)
                && bounds_ray_overlap(node.lower, node.upper, origin, translation)
            {
                if node.is_leaf() {
                    let child_idx = node.child_index as usize;
                    let child = &self.children[child_idx];
                    let inv_q = child.rotation.conjugate();
                    let local_origin = inv_q * (origin - child.position);
                    let local_dir = inv_q * ray_dir;
                    let ray = Ray::new(local_origin, local_dir);
                    let max_dist = fraction * trans_len;

                    if let Some((dist, local_normal)) = child.shape.ray_cast(&ray, max_dist) {
                        let alpha = dist / trans_len;
                        if alpha < fraction {
                            fraction = alpha;
                            best = Some(CompoundRayHit {
                                fraction: alpha,
                                point: origin + translation * alpha,
                                normal: child.rotation * local_normal,
                                child_index: child_idx,
                            });
                            (ray_lower, ray_upper) = bounds(fraction);
                        }
                    }
                } else if top + 2 <= stack.len() {
                    // Front child first
                    let (near, far) = if translation[node.axis as usize] > 0.0 {
                        (node.left, node.right)
                    } else {
                        (node.right, node.left)
                    };
                    stack[top] = far;
                    top += 1;
                    stack[top] = near;
                    top += 1;
                }
            }
        }

        best
    }
}

fn build_bvh(
    children: &[CompoundChild],
    indices: &mut [usize],
    nodes: &mut Vec<CompoundNode>,
) -> u32 {
    let count = indices.len();
    if count == 1 {
        let child_idx = indices[0];
        let (lower, upper) = children[child_idx].bounds();
        let node_id = nodes.len() as u32;
        nodes.push(CompoundNode {
            lower,
            upper,
            child_index: child_idx as u32,
            axis: 0,
            left: u32::MAX,
            right: u32::MAX,
        });
        return node_id;
    }

    let mut total_lower = Vec3::splat(f32::MAX);
    let mut total_upper = Vec3::splat(f32::MIN);
    let mut centers_min = Vec3::splat(f32::MAX);
    let mut centers_max = Vec3::splat(f32::MIN);

    for &idx in indices.iter() {
        let (lo, hi) = children[idx].bounds();
        total_lower = total_lower.min(lo);
        total_upper = total_upper.max(hi);
        let center = (lo + hi) * 0.5;
        centers_min = centers_min.min(center);
        centers_max = centers_max.max(center);
    }

    let span = centers_max - centers_min;
    let axis = if span.x >= span.y && span.x >= span.z {
        0u8
    } else if span.y >= span.z {
        1u8
    } else {
        2u8
    };

    let ax = axis as usize;
    indices.sort_by(|&a, &b| {
        let (lo_a, hi_a) = children[a].bounds();
        let (lo_b, hi_b) = children[b].bounds();
        let c_a = (lo_a[ax] + hi_a[ax]) * 0.5;
        let c_b = (lo_b[ax] + hi_b[ax]) * 0.5;
        c_a.partial_cmp(&c_b).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = count / 2;
    let left = build_bvh(children, &mut indices[..mid], nodes);
    let right = build_bvh(children, &mut indices[mid..], nodes);

    let node_id = nodes.len() as u32;
    nodes.push(CompoundNode {
        lower: total_lower,
        upper: total_upper,
        child_index: u32::MAX,
        axis,
        left,
        right,
    });
    node_id
}
