//! Box3D's GJK distance (`distance.c`) and the segment helpers of `math_functions.c`.
//!
//! [`shape_distance`] finds the closest points of two convex point clouds, shape B given in
//! shape A's frame, warm started from a [`SimplexCache`] that a contact keeps between steps.
//! With `use_radii` the shapes' radii are taken off the distance, and the closest points move
//! onto the rounded surfaces. The manifold builders use it for speculative contacts, sensors
//! for overlaps (`distance < OVERLAP_SLOP`). [`time_of_impact`] advances two [`Sweep`]s to the
//! first time the shapes come within a linear slop of touching, for continuous collision.

use crate::contact_solver::normalize;
use crate::math::{inv_rotate, lerp, nlerp, rotate, Transform};
use korppi_math::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Box3D's GJK iteration limit (`B3_MAX_GJK_ITERATIONS`).
const MAX_GJK_ITERATIONS: usize = 32;

/// Box3D's threshold for vectors too short to use, `1000 * FLT_MIN`.
const MIN_LENGTH_SQUARED: f32 = 1000.0 * f32::MIN_POSITIVE;

/// The distance below which Box3D treats shapes as overlapping (`B3_OVERLAP_SLOP`, a tenth of
/// the linear slop): GJK can return small positive distances for touching shapes.
pub const OVERLAP_SLOP: f32 = 0.1 * crate::contact_solver::LINEAR_SLOP;

/// A convex shape for GJK (Box3D `b3ShapeProxy`): the convex hull of `points`, rounded by
/// `radius`.
#[derive(Debug, Clone, Copy)]
pub struct ShapeProxy<'a> {
    /// The point cloud; at least one point.
    pub points: &'a [Vec3],
    /// The rounding radius.
    pub radius: f32,
}

/// The simplex GJK ended with, to start the next query from (Box3D `b3SimplexCache`). Zero
/// for a fresh query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SimplexCache {
    /// Length, area or volume of the simplex, to detect large changes.
    pub metric: f32,
    /// The number of cached vertices.
    pub count: u16,
    /// Vertex indices on shape A.
    pub index_a: [u8; 4],
    /// Vertex indices on shape B.
    pub index_b: [u8; 4],
}

/// The closest points of two shapes (Box3D `b3DistanceOutput`), in shape A's frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DistanceOutput {
    /// The closest point on shape A.
    pub point_a: Vec3,
    /// The closest point on shape B.
    pub point_b: Vec3,
    /// The unit direction from A to B; zero when the shapes overlap.
    pub normal: Vec3,
    /// The distance; zero when the shapes overlap.
    pub distance: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct SimplexVertex {
    w_a: Vec3,
    w_b: Vec3,
    w: Vec3,
    a: f32,
    index_a: usize,
    index_b: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct Simplex {
    v: [SimplexVertex; 4],
    count: usize,
}

/// Box3D `b3GetProxySupport`: the point farthest along `axis`, measured from the first point
/// for precision; the first of equals.
pub(crate) fn support_index(points: &[Vec3], axis: Vec3) -> usize {
    let origin = points[0];
    let mut best = 0;
    let mut max_projection = 0.0;
    for (index, &point) in points.iter().enumerate().skip(1) {
        let projection = axis.dot(point - origin);
        if projection > max_projection {
            best = index;
            max_projection = projection;
        }
    }
    best
}

fn blend2(s: f32, a: Vec3, t: f32, b: Vec3) -> Vec3 {
    Vec3::new(s * a.x + t * b.x, s * a.y + t * b.y, s * a.z + t * b.z)
}

fn blend3(s: f32, a: Vec3, t: f32, b: Vec3, u: f32, c: Vec3) -> Vec3 {
    Vec3::new(
        s * a.x + t * b.x + u * c.x,
        s * a.y + t * b.y + u * c.y,
        s * a.z + t * b.z + u * c.z,
    )
}

fn triple(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    a.dot(b.cross(c))
}

/// Box3D `b3BarycentricCoordsEdge`: unnormalized weights and the divisor.
fn edge_coords(a: Vec3, b: Vec3) -> [f32; 3] {
    let ab = b - a;
    [b.dot(ab), -a.dot(ab), ab.dot(ab)]
}

/// Box3D `b3BarycentricCoordsTri`.
fn triangle_coords(a: Vec3, b: Vec3, c: Vec3) -> [f32; 4] {
    let n = (b - a).cross(c - a);
    [
        b.cross(c).dot(n),
        c.cross(a).dot(n),
        a.cross(b).dot(n),
        n.dot(n),
    ]
}

/// Box3D `b3BarycentricCoordsTet`, with a positive divisor.
fn tetrahedron_coords(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> [f32; 5] {
    let divisor = triple(b - a, c - a, d - a);
    let sign = if divisor < 0.0 { -1.0 } else { 1.0 };
    [
        sign * triple(b, c, d),
        sign * triple(a, d, c),
        sign * triple(a, b, d),
        sign * triple(a, c, b),
        sign * divisor,
    ]
}

impl Simplex {
    fn metric(&self) -> f32 {
        let v = &self.v;
        match self.count {
            2 => (v[1].w - v[0].w).length(),
            3 => (v[1].w - v[0].w).cross(v[2].w - v[0].w).length() / 2.0,
            4 => triple(v[1].w - v[0].w, v[2].w - v[0].w, v[3].w - v[0].w) / 6.0,
            _ => 0.0,
        }
    }

    fn write_cache(&self, cache: &mut SimplexCache) {
        cache.metric = self.metric();
        cache.count = self.count as u16;
        for (i, v) in self.v[..self.count].iter().enumerate() {
            cache.index_a[i] = v.index_a as u8;
            cache.index_b[i] = v.index_b as u8;
        }
    }

    /// Keeps the vertices `keep` with weights `weights / divisor`; fails on a zero divisor.
    fn reduce(&mut self, keep: &[SimplexVertex], weights: &[f32], divisor: f32) -> bool {
        if divisor <= 0.0 {
            return false;
        }
        for (i, (&vertex, &weight)) in keep.iter().zip(weights).enumerate() {
            self.v[i] = vertex;
            self.v[i].a = weight / divisor;
        }
        self.count = keep.len();
        true
    }

    fn keep_vertex(&mut self, vertex: SimplexVertex) -> bool {
        self.v[0] = vertex;
        self.v[0].a = 1.0;
        self.count = 1;
        true
    }

    /// Box3D `b3SolveSimplex2`.
    fn solve2(&mut self) -> bool {
        let (a, b) = (self.v[0], self.v[1]);
        let [u, v, divisor] = edge_coords(a.w, b.w);
        if v <= 0.0 {
            return self.keep_vertex(a);
        }
        if u <= 0.0 {
            return self.keep_vertex(b);
        }
        if divisor <= 0.0 {
            return false;
        }
        let inv = 1.0 / divisor;
        self.v[0].a = inv * u;
        self.v[1].a = inv * v;
        true
    }

    /// Box3D `b3SolveSimplex3`.
    fn solve3(&mut self) -> bool {
        let (v1, v2, v3) = (self.v[0], self.v[1], self.v[2]);
        let w_ab = edge_coords(v1.w, v2.w);
        let w_bc = edge_coords(v2.w, v3.w);
        let w_ca = edge_coords(v3.w, v1.w);
        if w_ab[1] <= 0.0 && w_ca[0] <= 0.0 {
            return self.keep_vertex(v1);
        }
        if w_bc[1] <= 0.0 && w_ab[0] <= 0.0 {
            return self.keep_vertex(v2);
        }
        if w_ca[1] <= 0.0 && w_bc[0] <= 0.0 {
            return self.keep_vertex(v3);
        }
        let w_abc = triangle_coords(v1.w, v2.w, v3.w);
        if w_abc[2] <= 0.0 && w_ab[0] > 0.0 && w_ab[1] > 0.0 {
            return self.reduce(&[v1, v2], &w_ab[..2], w_ab[2]);
        }
        if w_abc[0] <= 0.0 && w_bc[0] > 0.0 && w_bc[1] > 0.0 {
            return self.reduce(&[v2, v3], &w_bc[..2], w_bc[2]);
        }
        if w_abc[1] <= 0.0 && w_ca[0] > 0.0 && w_ca[1] > 0.0 {
            return self.reduce(&[v3, v1], &w_ca[..2], w_ca[2]);
        }
        self.reduce(&[v1, v2, v3], &w_abc[..3], w_abc[3])
    }

    /// Box3D `b3SolveSimplex4`.
    fn solve4(&mut self) -> bool {
        let (a, b, c, d) = (self.v[0], self.v[1], self.v[2], self.v[3]);
        let w_ab = edge_coords(a.w, b.w);
        let w_ac = edge_coords(a.w, c.w);
        let w_ad = edge_coords(a.w, d.w);
        let w_bc = edge_coords(b.w, c.w);
        let w_cd = edge_coords(c.w, d.w);
        let w_db = edge_coords(d.w, b.w);
        if w_ab[1] <= 0.0 && w_ac[1] <= 0.0 && w_ad[1] <= 0.0 {
            return self.keep_vertex(a);
        }
        if w_ab[0] <= 0.0 && w_db[0] <= 0.0 && w_bc[1] <= 0.0 {
            return self.keep_vertex(b);
        }
        if w_ac[0] <= 0.0 && w_bc[0] <= 0.0 && w_cd[1] <= 0.0 {
            return self.keep_vertex(c);
        }
        if w_ad[0] <= 0.0 && w_cd[0] <= 0.0 && w_db[1] <= 0.0 {
            return self.keep_vertex(d);
        }
        let w_acb = triangle_coords(a.w, c.w, b.w);
        let w_abd = triangle_coords(a.w, b.w, d.w);
        let w_adc = triangle_coords(a.w, d.w, c.w);
        let w_bcd = triangle_coords(b.w, c.w, d.w);
        if w_abd[2] <= 0.0 && w_acb[1] <= 0.0 && w_ab[0] > 0.0 && w_ab[1] > 0.0 {
            return self.reduce(&[a, b], &w_ab[..2], w_ab[2]);
        }
        if w_acb[2] <= 0.0 && w_adc[1] <= 0.0 && w_ac[0] > 0.0 && w_ac[1] > 0.0 {
            return self.reduce(&[a, c], &w_ac[..2], w_ac[2]);
        }
        if w_adc[2] <= 0.0 && w_abd[1] <= 0.0 && w_ad[0] > 0.0 && w_ad[1] > 0.0 {
            return self.reduce(&[a, d], &w_ad[..2], w_ad[2]);
        }
        if w_acb[0] <= 0.0 && w_bcd[2] <= 0.0 && w_bc[0] > 0.0 && w_bc[1] > 0.0 {
            return self.reduce(&[b, c], &w_bc[..2], w_bc[2]);
        }
        if w_adc[0] <= 0.0 && w_bcd[0] <= 0.0 && w_cd[0] > 0.0 && w_cd[1] > 0.0 {
            return self.reduce(&[c, d], &w_cd[..2], w_cd[2]);
        }
        if w_abd[0] <= 0.0 && w_bcd[1] <= 0.0 && w_db[0] > 0.0 && w_db[1] > 0.0 {
            return self.reduce(&[d, b], &w_db[..2], w_db[2]);
        }
        let w_abcd = tetrahedron_coords(a.w, b.w, c.w, d.w);
        let positive = |w: &[f32; 4]| w[0] > 0.0 && w[1] > 0.0 && w[2] > 0.0;
        if w_abcd[3] < 0.0 && positive(&w_acb) {
            return self.reduce(&[a, c, b], &w_acb[..3], w_acb[3]);
        }
        if w_abcd[2] < 0.0 && positive(&w_abd) {
            return self.reduce(&[a, b, d], &w_abd[..3], w_abd[3]);
        }
        if w_abcd[1] < 0.0 && positive(&w_adc) {
            return self.reduce(&[a, d, c], &w_adc[..3], w_adc[3]);
        }
        if w_abcd[0] < 0.0 && positive(&w_bcd) {
            return self.reduce(&[b, c, d], &w_bcd[..3], w_bcd[3]);
        }
        self.reduce(&[a, b, c, d], &w_abcd[..4], w_abcd[4])
    }

    /// Box3D `b3ComputeWitnessPoints`.
    fn witness_points(&self) -> (Vec3, Vec3) {
        let v = &self.v;
        match self.count {
            1 => (v[0].w_a, v[0].w_b),
            2 => (
                blend2(v[0].a, v[0].w_a, v[1].a, v[1].w_a),
                blend2(v[0].a, v[0].w_b, v[1].a, v[1].w_b),
            ),
            3 => (
                blend3(v[0].a, v[0].w_a, v[1].a, v[1].w_a, v[2].a, v[2].w_a),
                blend3(v[0].a, v[0].w_b, v[1].a, v[1].w_b, v[2].a, v[2].w_b),
            ),
            _ => {
                // Identical points and zero distance.
                let sum = blend2(v[0].a, v[0].w_a, v[1].a, v[1].w_a)
                    + blend2(v[2].a, v[2].w_a, v[3].a, v[3].w_a);
                (sum, sum)
            }
        }
    }

    fn closest_point(&self) -> Vec3 {
        let v = &self.v;
        match self.count {
            1 => v[0].w,
            2 => blend2(v[0].a, v[0].w, v[1].a, v[1].w),
            3 => blend3(v[0].a, v[0].w, v[1].a, v[1].w, v[2].a, v[2].w),
            _ => blend2(v[0].a, v[0].w, v[1].a, v[1].w) + blend2(v[2].a, v[2].w, v[3].a, v[3].w),
        }
    }
}

/// Box3D `b3ShapeDistance`: the closest points of `proxy_a` and `proxy_b`, B placed in A's
/// frame by `transform`, starting from and updating `cache`. With `use_radii` the radii are
/// subtracted from the distance and the points moved onto the rounded surfaces.
pub fn shape_distance(
    proxy_a: &ShapeProxy<'_>,
    proxy_b: &ShapeProxy<'_>,
    transform: &Transform,
    use_radii: bool,
    cache: &mut SimplexCache,
) -> DistanceOutput {
    let m = Mat3::from_quat(transform.q);
    let mt = m.transpose();
    let point_b = |index: usize| m * proxy_b.points[index] + transform.p;

    let mut simplex = Simplex {
        count: cache.count as usize,
        ..Default::default()
    };
    for i in 0..simplex.count {
        let (index_a, index_b) = (cache.index_a[i] as usize, cache.index_b[i] as usize);
        if index_a >= proxy_a.points.len() || index_b >= proxy_b.points.len() {
            simplex.count = 0;
            break;
        }
        let w_a = proxy_a.points[index_a];
        let w_b = point_b(index_b);
        simplex.v[i] = SimplexVertex {
            w_a,
            w_b,
            w: w_b - w_a,
            a: 0.0,
            index_a,
            index_b,
        };
    }

    // Flush a simplex whose size changed substantially.
    if simplex.count > 0 {
        let metric1 = cache.metric;
        let metric2 = simplex.metric();
        if 2.0 * metric1 < metric2 || metric2 < 0.5 * metric1 || metric2 < f32::EPSILON {
            simplex.count = 0;
        }
    }

    if simplex.count == 0 {
        let w_a = proxy_a.points[0];
        let w_b = point_b(0);
        simplex.v[0] = SimplexVertex {
            w_a,
            w_b,
            w: w_b - w_a,
            ..Default::default()
        };
        simplex.count = 1;
    }

    let overlap = |simplex: &Simplex| {
        let (point_a, point_b) = simplex.witness_points();
        DistanceOutput {
            point_a,
            point_b,
            normal: Vec3::ZERO,
            distance: 0.0,
        }
    };

    let mut backup = Simplex::default();
    let mut distance_squared = f32::MAX;
    let mut normal = Vec3::ZERO;

    for _ in 0..MAX_GJK_ITERATIONS {
        let solved = match simplex.count {
            1 => {
                simplex.v[0].a = 1.0;
                true
            }
            2 => simplex.solve2(),
            3 => simplex.solve3(),
            _ => simplex.solve4(),
        };
        if !solved {
            // No progress: back to the last simplex.
            simplex = backup;
            break;
        }

        if simplex.count == 4 {
            return overlap(&simplex);
        }

        let old_distance_squared = distance_squared;
        let closest = simplex.closest_point();
        distance_squared = closest.dot(closest);
        if distance_squared >= old_distance_squared {
            simplex = backup;
            break;
        }

        let v = &simplex.v;
        let search_direction = match simplex.count {
            1 => -v[0].w,
            2 => {
                let ab = v[1].w - v[0].w;
                ab.cross(-v[0].w).cross(ab)
            }
            _ => {
                let n = (v[1].w - v[0].w).cross(v[2].w - v[0].w);
                if n.dot(v[0].w) < 0.0 {
                    n
                } else {
                    -n
                }
            }
        };

        if search_direction.length_squared() < MIN_LENGTH_SQUARED {
            // The origin is on a segment or triangle: the shapes overlap.
            return overlap(&simplex);
        }

        normal = -search_direction;

        let index_a = support_index(proxy_a.points, -search_direction);
        let support_a = proxy_a.points[index_a];
        let index_b = support_index(proxy_b.points, mt * search_direction);
        let support_b = point_b(index_b);

        backup = simplex;

        // Duplicate support points end the search.
        if simplex.v[..simplex.count]
            .iter()
            .any(|v| v.index_a == index_a && v.index_b == index_b)
        {
            break;
        }

        simplex.v[simplex.count] = SimplexVertex {
            w_a: support_a,
            w_b: support_b,
            w: support_b - support_a,
            a: 0.0,
            index_a,
            index_b,
        };
        simplex.count += 1;
    }

    let (point_a, point_b) = simplex.witness_points();
    let normal = normalize(normal);
    if (1.0 - normal.dot(normal)).abs() >= 100.0 * f32::EPSILON {
        // Treat as overlap.
        return DistanceOutput {
            point_a,
            point_b,
            normal: Vec3::ZERO,
            distance: 0.0,
        };
    }

    let mut output = DistanceOutput {
        point_a,
        point_b,
        normal,
        distance: (point_b - point_a).length(),
    };

    if use_radii {
        let (r_a, r_b) = (proxy_a.radius, proxy_b.radius);
        output.distance = (output.distance - r_a - r_b).max(0.0);
        // Keep the points on the surfaces even when overlapped, so they move smoothly.
        output.point_a += normal * r_a;
        output.point_b -= normal * r_b;
    }

    simplex.write_cache(cache);
    output
}

/// The closest points of two segments or lines (Box3D `b3SegmentDistanceResult`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentDistance {
    /// Closest point on the first segment or line.
    pub point1: Vec3,
    /// Closest point on the second segment or line.
    pub point2: Vec3,
    /// Normalized fraction along the first segment `[0.0, 1.0]`.
    pub fraction1: f32,
    /// Normalized fraction along the second segment `[0.0, 1.0]`.
    pub fraction2: f32,
}

impl SegmentDistance {
    /// Box3D `b3IsWithinSegments`.
    pub fn is_within_segments(&self) -> bool {
        (0.0..=1.0).contains(&self.fraction1) && (0.0..=1.0).contains(&self.fraction2)
    }
}

/// Box3D `b3LineDistance`: the closest points of the lines `p1 + s d1` and `p2 + s d2`.
pub(crate) fn line_distance(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> SegmentDistance {
    let a11 = d1.dot(d1);
    let a12 = -d1.dot(d2);
    let a21 = d2.dot(d1);
    let a22 = -d2.dot(d2);
    let w = p1 - p2;
    let b1 = -d1.dot(w);
    let b2 = -d2.dot(w);
    let det = a11 * a22 - a12 * a21;
    let (s1, s2) = if det * det < MIN_LENGTH_SQUARED {
        // Parallel: project p2 onto the first line.
        ((p2 - p1).dot(d1) / d1.dot(d1), 0.0)
    } else {
        ((a22 * b1 - a12 * b2) / det, (a11 * b2 - a21 * b1) / det)
    };
    SegmentDistance {
        point1: p1 + d1 * s1,
        point2: p2 + d2 * s2,
        fraction1: s1,
        fraction2: s2,
    }
}

/// Box3D `b3SegmentDistance`: the closest points of segments `p1 q1` and `p2 q2`.
pub fn segment_distance(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> SegmentDistance {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let b = d1.dot(d2);
    let c = d1.dot(r);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let tiny = 100.0 * f32::EPSILON;
    let result = |s1: f32, s2: f32| SegmentDistance {
        point1: p1 + d1 * s1,
        point2: p2 + d2 * s2,
        fraction1: s1,
        fraction2: s2,
    };
    if a < tiny && e < tiny {
        return SegmentDistance {
            point1: p1,
            point2: p2,
            fraction1: 0.0,
            fraction2: 0.0,
        };
    }
    if a < tiny {
        let s2 = (f / e).clamp(0.0, 1.0);
        return SegmentDistance {
            point1: p1,
            fraction1: 0.0,
            ..result(0.0, s2)
        };
    }
    if e < tiny {
        let s1 = (-c / a).clamp(0.0, 1.0);
        return SegmentDistance {
            point2: p2,
            fraction2: 0.0,
            ..result(s1, 0.0)
        };
    }
    let denominator = a * e - b * b;
    let mut s1 = if denominator > MIN_LENGTH_SQUARED {
        ((b * f - c * e) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut s2 = (b * s1 + f) / e;
    if s2 < 0.0 {
        s1 = (-c / a).clamp(0.0, 1.0);
        s2 = 0.0;
    } else if s2 > 1.0 {
        s1 = ((b - c) / a).clamp(0.0, 1.0);
        s2 = 1.0;
    }
    result(s1, s2)
}

/// Box3D `b3PointToSegmentDistance`: the point of segment `a b` closest to `q`.
pub fn closest_point_on_segment(a: Vec3, b: Vec3, q: Vec3) -> Vec3 {
    let ab = b - a;
    let alpha = ab.dot(q - a);
    if alpha <= 0.0 {
        return a;
    }
    let denominator = ab.dot(ab);
    if alpha > denominator {
        b
    } else {
        a + ab * (alpha / denominator)
    }
}

/// A body's motion over a step for the time of impact (Box3D `b3Sweep`): its centre of mass
/// from `c1` to `c2` and its rotation from `q1` to `q2`; `local_center` is the centre of mass
/// in the body's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Sweep {
    pub local_center: Vec3,
    pub c1: Vec3,
    pub c2: Vec3,
    pub q1: Quat,
    pub q2: Quat,
}

impl Sweep {
    /// A body at rest at `transform` (centre of mass at the origin).
    pub fn at_rest(transform: &Transform) -> Self {
        Self {
            local_center: Vec3::ZERO,
            c1: transform.p,
            c2: transform.p,
            q1: transform.q,
            q2: transform.q,
        }
    }

    /// The sweep of a shape placed on the body by `frame`: the same motion of the centre of
    /// mass, with the frame's rotation and the centre of mass in the shape's frame.
    pub fn of_frame(&self, frame: &crate::joint::Frame3d) -> Self {
        Self {
            local_center: frame.rotation.conjugate() * (self.local_center - frame.position),
            q1: self.q1 * frame.rotation,
            q2: self.q2 * frame.rotation,
            ..*self
        }
    }

    /// Box3D `b3GetSweepTransform`: the body's pose at `time` in [0, 1].
    pub fn transform(&self, time: f32) -> Transform {
        let q = nlerp(self.q1, self.q2, time);
        Transform::new(
            lerp(self.c1, self.c2, time) - rotate(q, self.local_center),
            q,
        )
    }

    fn final_transform(&self) -> Transform {
        Transform::new(self.c2 - rotate(self.q2, self.local_center), self.q2)
    }
}

/// How a time of impact query ended (Box3D `b3TOIState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToiState {
    Failed,
    Overlapped,
    Hit,
    Separated,
}

/// The result of a time of impact query (Box3D `b3TOIOutput`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ToiOutput {
    pub state: ToiState,
    /// The fraction of the sweep at the impact; zero when the shapes start overlapped.
    pub fraction: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeparationKind {
    Vertices,
    Edges,
    FaceA,
    FaceB,
}

/// Box3D `b3SeparationFunction`: the separation of the shapes along an axis that moves with
/// them, from the GJK simplex.
struct SeparationFunction<'a> {
    proxy_a: &'a ShapeProxy<'a>,
    proxy_b: &'a ShapeProxy<'a>,
    sweep_a: Sweep,
    sweep_b: Sweep,
    witness1: Vec3,
    witness2: Vec3,
    kind: SeparationKind,
}

/// Box3D `b3UniqueCount`.
fn unique_count(count: usize, v: &[usize; 3]) -> usize {
    match count {
        1 => 1,
        2 => {
            if v[0] == v[1] {
                1
            } else {
                2
            }
        }
        _ => {
            if v[0] != v[1] && v[0] != v[2] && v[1] != v[2] {
                3
            } else if v[0] == v[1] && v[0] == v[2] {
                1
            } else {
                2
            }
        }
    }
}

/// Box3D `b3CheckFastEdges`: whether the cross product of two edges flips over the sweep.
fn edges_flip(xf_a: &Transform, edge_a: Vec3, xf_b: &Transform, edge_b: Vec3, axis0: Vec3) -> bool {
    rotate(xf_a.q, edge_a)
        .cross(rotate(xf_b.q, edge_b))
        .dot(axis0)
        < 0.0
}

impl<'a> SeparationFunction<'a> {
    /// Box3D `b3MakeSeparationFunction`.
    fn new(
        cache: &SimplexCache,
        proxy_a: &'a ShapeProxy<'a>,
        sweep_a: &Sweep,
        proxy_b: &'a ShapeProxy<'a>,
        sweep_b: &Sweep,
        world_normal: Vec3,
        t1: f32,
    ) -> Self {
        let mut f = Self {
            proxy_a,
            proxy_b,
            sweep_a: *sweep_a,
            sweep_b: *sweep_b,
            witness1: world_normal,
            witness2: Vec3::ZERO,
            kind: SeparationKind::Vertices,
        };
        let count = cache.count as usize;
        let mut index_a = [0usize; 3];
        let mut index_b = [0usize; 3];
        for i in 0..count.min(3) {
            index_a[i] = cache.index_a[i] as usize;
            index_b[i] = cache.index_b[i] as usize;
        }
        let unique_a = unique_count(count, &index_a);
        let unique_b = unique_count(count, &index_b);
        let xf_a1 = sweep_a.transform(t1);
        let xf_b1 = sweep_b.transform(t1);
        let (q_a, q_b) = (xf_a1.q, xf_b1.q);
        let delta_p = xf_b1.p - xf_a1.p;
        let points_a = proxy_a.points;
        let points_b = proxy_b.points;

        // An edge pair, or the world normal if the edges are nearly parallel or their cross
        // product flips over the sweep.
        let edge_pair =
            |f: &mut Self, (a1, a2): (usize, usize), (b1, b2): (usize, usize), tolerance: f32| {
                let v_a1 = points_a[a1];
                let local_edge_a = normalize(points_a[a2] - v_a1);
                let edge_a = rotate(q_a, local_edge_a);
                let v_b1 = points_b[b1];
                let mut local_edge_b = normalize(points_b[b2] - v_b1);
                let edge_b = rotate(q_b, local_edge_b);
                let mut axis = edge_a.cross(edge_b);
                if axis.length_squared() < tolerance * tolerance {
                    return;
                }
                let delta = rotate(q_b, v_b1) - rotate(q_a, v_a1) + delta_p;
                if delta.dot(axis) < 0.0 {
                    // Make the axis point from A to B.
                    axis = -axis;
                    local_edge_b = -local_edge_b;
                }
                if edges_flip(
                    &sweep_a.final_transform(),
                    local_edge_a,
                    &sweep_b.final_transform(),
                    local_edge_b,
                    axis,
                ) {
                    f.kind = SeparationKind::Vertices;
                    f.witness1 = normalize(axis);
                } else {
                    f.kind = SeparationKind::Edges;
                    f.witness1 = local_edge_a;
                    f.witness2 = local_edge_b;
                }
            };

        match count {
            2 if unique_a == 2 && unique_b == 2 => {
                edge_pair(
                    &mut f,
                    (index_a[0], index_a[1]),
                    (index_b[0], index_b[1]),
                    0.05,
                );
            }
            3 if unique_a == 3 => {
                let (v1, v2, v3) = (
                    points_a[index_a[0]],
                    points_a[index_a[1]],
                    points_a[index_a[2]],
                );
                let mut local_axis = normalize((v2 - v1).cross(v3 - v1));
                let local_point_a = (v1 + v2 + v3) * (1.0 / 3.0);
                let local_point_b = points_b[index_b[0]];
                let delta = rotate(q_b, local_point_b) - rotate(q_a, local_point_a) + delta_p;
                if delta.dot(rotate(q_a, local_axis)) < 0.0 {
                    local_axis = -local_axis;
                }
                f.kind = SeparationKind::FaceA;
                f.witness1 = local_axis;
                f.witness2 = local_point_a;
            }
            3 if unique_b == 3 => {
                let (v1, v2, v3) = (
                    points_b[index_b[0]],
                    points_b[index_b[1]],
                    points_b[index_b[2]],
                );
                let mut local_axis = normalize((v2 - v1).cross(v3 - v1));
                let local_point_a = points_a[index_a[0]];
                let local_point_b = (v1 + v2 + v3) * (1.0 / 3.0);
                let delta = rotate(q_a, local_point_a) - rotate(q_b, local_point_b) - delta_p;
                if delta.dot(rotate(q_b, local_axis)) < 0.0 {
                    local_axis = -local_axis;
                }
                f.kind = SeparationKind::FaceB;
                f.witness1 = local_axis;
                f.witness2 = local_point_b;
            }
            3 => {
                // Two unique vertices on each side: an edge pair.
                if index_a[0] == index_a[1] {
                    index_a[1] = index_a[2];
                }
                if index_b[0] == index_b[1] {
                    index_b[1] = index_b[2];
                }
                edge_pair(
                    &mut f,
                    (index_a[0], index_a[1]),
                    (index_b[0], index_b[1]),
                    0.005,
                );
            }
            _ => {}
        }
        f
    }

    /// Box3D `b3FindMinSeparation`: the deepest points along the axis at time `t`.
    fn find_min_separation(&self, t: f32) -> (f32, usize, usize) {
        let xf_a = self.sweep_a.transform(t);
        let xf_b = self.sweep_b.transform(t);
        let (points_a, points_b) = (self.proxy_a.points, self.proxy_b.points);
        let delta_p = xf_b.p - xf_a.p;
        let along = |axis: Vec3| {
            let index_a = support_index(points_a, inv_rotate(xf_a.q, axis));
            let index_b = support_index(points_b, inv_rotate(xf_b.q, -axis));
            let delta =
                rotate(xf_b.q, points_b[index_b]) - rotate(xf_a.q, points_a[index_a]) + delta_p;
            (delta.dot(axis), index_a, index_b)
        };
        match self.kind {
            SeparationKind::Vertices => along(self.witness1),
            SeparationKind::Edges => {
                let axis =
                    normalize(rotate(xf_a.q, self.witness1).cross(rotate(xf_b.q, self.witness2)));
                along(axis)
            }
            SeparationKind::FaceA => {
                let normal = rotate(xf_a.q, self.witness1);
                let point_a = xf_a.apply(self.witness2);
                let index_b = support_index(points_b, inv_rotate(xf_b.q, -normal));
                let point_b = xf_b.apply(points_b[index_b]);
                ((point_b - point_a).dot(normal), 0, index_b)
            }
            SeparationKind::FaceB => {
                let normal = rotate(xf_b.q, self.witness1);
                let index_a = support_index(points_a, inv_rotate(xf_a.q, -normal));
                let point_a = xf_a.apply(points_a[index_a]);
                let point_b = xf_b.apply(self.witness2);
                ((point_a - point_b).dot(normal), index_a, 0)
            }
        }
    }

    /// Box3D `b3EvaluateSeparation`: the separation of the given points at time `t`.
    fn evaluate(&self, index_a: usize, index_b: usize, t: f32) -> f32 {
        let xf_a = self.sweep_a.transform(t);
        let xf_b = self.sweep_b.transform(t);
        let (points_a, points_b) = (self.proxy_a.points, self.proxy_b.points);
        match self.kind {
            SeparationKind::Vertices => {
                (xf_b.apply(points_b[index_b]) - xf_a.apply(points_a[index_a])).dot(self.witness1)
            }
            SeparationKind::Edges => {
                let axis =
                    normalize(rotate(xf_a.q, self.witness1).cross(rotate(xf_b.q, self.witness2)));
                (xf_b.apply(points_b[index_b]) - xf_a.apply(points_a[index_a])).dot(axis)
            }
            SeparationKind::FaceA => {
                let axis = rotate(xf_a.q, self.witness1);
                (xf_b.apply(points_b[index_b]) - xf_a.apply(self.witness2)).dot(axis)
            }
            SeparationKind::FaceB => {
                let axis = rotate(xf_b.q, self.witness1);
                (xf_a.apply(points_a[index_a]) - xf_b.apply(self.witness2)).dot(axis)
            }
        }
    }

    /// Box3D `b3ForceFixedAxis`: an edge axis frozen at time `t`.
    fn force_fixed_axis(&mut self, t: f32) {
        let xf_a = self.sweep_a.transform(t);
        let xf_b = self.sweep_b.transform(t);
        self.witness1 =
            normalize(rotate(xf_a.q, self.witness1).cross(rotate(xf_b.q, self.witness2)));
        self.witness2 = Vec3::ZERO;
        self.kind = SeparationKind::Vertices;
    }
}

/// Box3D `b3TimeOfImpact`: the first fraction of the sweeps, up to `max_fraction`, at which
/// the shapes come within a linear slop of touching, by conservative advancement on separating
/// axes found with GJK.
pub(crate) fn time_of_impact(
    proxy_a: &ShapeProxy<'_>,
    sweep_a: &Sweep,
    proxy_b: &ShapeProxy<'_>,
    sweep_b: &Sweep,
    max_fraction: f32,
) -> ToiOutput {
    // Shift to the origin for precision.
    let origin = sweep_a.c1;
    let mut sweep_a = *sweep_a;
    let mut sweep_b = *sweep_b;
    sweep_a.c1 = Vec3::ZERO;
    sweep_a.c2 -= origin;
    sweep_b.c1 -= origin;
    sweep_b.c2 -= origin;

    let max_push_back_iterations = proxy_a.points.len() + proxy_b.points.len();
    let t_max = max_fraction;
    let slop = crate::contact_solver::LINEAR_SLOP;
    let total_radius = proxy_a.radius + proxy_b.radius;
    let target = slop.max(total_radius - slop);
    let tolerance = 0.25 * slop;
    // The shapes' cores; the radii are in the target.
    let core_a = ShapeProxy {
        radius: 0.0,
        ..*proxy_a
    };
    let core_b = ShapeProxy {
        radius: 0.0,
        ..*proxy_b
    };

    let mut t1 = 0.0;
    let max_iterations = 25;
    let mut distance_iterations = 0;
    let mut cache = SimplexCache::default();

    loop {
        let xf_a = sweep_a.transform(t1);
        let xf_b = sweep_b.transform(t1);
        let output = shape_distance(&core_a, &core_b, &xf_a.inv_mul(&xf_b), false, &mut cache);
        let world_normal = rotate(xf_a.q, output.normal);
        distance_iterations += 1;

        if output.distance <= 0.0 {
            return ToiOutput {
                state: ToiState::Overlapped,
                fraction: 0.0,
            };
        }
        if output.distance <= target + tolerance {
            return ToiOutput {
                state: ToiState::Hit,
                fraction: t1,
            };
        }
        if distance_iterations == max_iterations {
            // Too slow, as when a capsule rotates around a triangle vertex.
            return ToiOutput {
                state: ToiState::Failed,
                fraction: t1,
            };
        }

        let mut function = SeparationFunction::new(
            &cache,
            &core_a,
            &sweep_a,
            &core_b,
            &sweep_b,
            world_normal,
            t1,
        );

        // Resolve the deepest point along the axis until it is within tolerance.
        let mut t2 = t_max;
        let mut push_back_iterations = 0;
        loop {
            let (mut s2, index_a, index_b) = function.find_min_separation(t2);

            if s2 - target > tolerance {
                // Separated at the end of the sweep.
                return ToiOutput {
                    state: ToiState::Separated,
                    fraction: max_fraction,
                };
            }
            if s2 >= target - tolerance {
                // Advance the sweeps.
                t1 = t2;
                break;
            }

            let mut s1 = function.evaluate(index_a, index_b, t1);
            if s1 < target - tolerance {
                return ToiOutput {
                    state: ToiState::Failed,
                    fraction: t1,
                };
            }
            if s1 <= target + tolerance {
                return ToiOutput {
                    state: ToiState::Hit,
                    fraction: t1,
                };
            }

            // The root of s(t) = target, by false position and bisection.
            let mut root_iterations = 0;
            let max_root_iterations = 50;
            let (mut a1, mut a2) = (t1, t2);
            loop {
                let t = if root_iterations & 1 == 1 {
                    a1 + (target - s1) * (a2 - a1) / (s2 - s1)
                } else {
                    0.5 * (a1 + a2)
                };
                root_iterations += 1;
                let s = function.evaluate(index_a, index_b, t);
                if (s - target).abs() <= tolerance {
                    t2 = t;
                    break;
                }
                if s > target {
                    a1 = t;
                    s1 = s;
                } else {
                    a2 = t;
                    s2 = s;
                }
                if root_iterations == max_root_iterations {
                    break;
                }
            }

            // Restart with a fixed axis if an edge axis fails to converge.
            if root_iterations == max_root_iterations - 1 && function.kind == SeparationKind::Edges
            {
                t2 = max_fraction;
                function.force_fixed_axis(t1);
            }

            push_back_iterations += 1;
            if push_back_iterations == max_push_back_iterations {
                break;
            }
        }
    }
}

/// Output of a shape-cast query (Box3D `b3CastOutput`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CastOutput {
    /// Normal at the hit point in shape A's frame, pointing from shape A toward shape B.
    pub normal: Vec3,
    /// Hit point on the surface of shape A in frame A.
    pub point: Vec3,
    /// Hit fraction of the input translation in `[0.0, max_fraction]`.
    pub fraction: f32,
    /// Number of GJK iterations performed.
    pub iterations: u32,
    /// Whether a hit occurred.
    pub hit: bool,
}

impl Default for CastOutput {
    fn default() -> Self {
        Self {
            normal: Vec3::ZERO,
            point: Vec3::ZERO,
            fraction: 0.0,
            iterations: 0,
            hit: false,
        }
    }
}

/// Sweeps shape B along `translation_b` against stationary shape A using conservative
/// advancement (Box3D `b3ShapeCast`).
///
/// Returns the hit fraction, surface point, and contact normal in shape A's frame.
pub fn shape_cast(
    proxy_a: &ShapeProxy<'_>,
    proxy_b: &ShapeProxy<'_>,
    transform: &Transform,
    translation_b: Vec3,
    max_fraction: f32,
    can_encroach: bool,
) -> CastOutput {
    let linear_slop = crate::contact_solver::LINEAR_SLOP;
    let total_radius = proxy_a.radius + proxy_b.radius;
    let mut target = linear_slop.max(total_radius - linear_slop);
    let tolerance = 0.25 * linear_slop;

    let mut cache = SimplexCache::default();
    let mut alpha = 0.0f32;

    let mut current_transform = *transform;
    let delta2 = translation_b;
    let mut output = CastOutput::default();

    for iteration in 0..20 {
        output.iterations += 1;
        let distance_output =
            shape_distance(proxy_a, proxy_b, &current_transform, false, &mut cache);

        if distance_output.distance < target + tolerance {
            if iteration == 0 {
                if can_encroach && distance_output.distance > 2.0 * linear_slop {
                    target = distance_output.distance - linear_slop;
                } else {
                    // Initial overlap
                    output.hit = true;
                    let c1 = distance_output.point_a + distance_output.normal * proxy_a.radius;
                    let c2 = distance_output.point_b - distance_output.normal * proxy_b.radius;
                    output.point = (c1 + c2) * 0.5;
                    output.normal = distance_output.normal;
                    return output;
                }
            } else {
                output.fraction = alpha;
                output.point = distance_output.point_a + distance_output.normal * proxy_a.radius;
                output.normal = distance_output.normal;
                output.hit = true;
                return output;
            }
        }

        let denominator = delta2.dot(distance_output.normal);
        if denominator >= 0.0 {
            // Miss
            return output;
        }

        alpha += (target - distance_output.distance) / denominator;
        if alpha >= max_fraction {
            return output;
        }

        current_transform.p = transform.p + delta2 * alpha;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use korppi_math::Quat;

    fn cube() -> [Vec3; 8] {
        let h = 0.5;
        [
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
        ]
    }

    #[test]
    fn separated_shapes_report_their_closest_points() {
        let box_points = cube();
        let a = ShapeProxy {
            points: &box_points,
            radius: 0.0,
        };
        let center = [Vec3::ZERO];
        let ball = ShapeProxy {
            points: &center,
            radius: 0.25,
        };
        let mut cache = SimplexCache::default();
        let xf = Transform::new(Vec3::new(0.0, 1.5, 0.2), Quat::IDENTITY);
        let out = shape_distance(&a, &ball, &xf, true, &mut cache);
        assert!((out.distance - 0.75).abs() < 1e-6, "{out:?}");
        assert!((out.normal - Vec3::Y).length() < 1e-6);
        assert!((out.point_a - Vec3::new(0.0, 0.5, 0.2)).length() < 1e-6);
        assert!((out.point_b - Vec3::new(0.0, 1.25, 0.2)).length() < 1e-6);
        assert!(cache.count > 0, "the simplex is cached");

        // Warm started from the cache, the same answer.
        let again = shape_distance(&a, &ball, &xf, true, &mut cache);
        assert!((again.distance - out.distance).abs() < 1e-6);

        // Two boxes, one turned, 0.3 m apart along x.
        let b = ShapeProxy {
            points: &box_points,
            radius: 0.0,
        };
        let xf = Transform::new(Vec3::new(1.3, 0.0, 0.0), Quat::from_rotation_y(0.0));
        let out = shape_distance(&a, &b, &xf, false, &mut SimplexCache::default());
        assert!((out.distance - 0.3).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn overlapping_shapes_have_zero_distance() {
        let box_points = cube();
        let a = ShapeProxy {
            points: &box_points,
            radius: 0.0,
        };
        let xf = Transform::new(Vec3::new(0.5, 0.2, 0.0), Quat::from_rotation_z(0.3));
        let out = shape_distance(&a, &a, &xf, false, &mut SimplexCache::default());
        assert_eq!(out.distance, 0.0);
        assert_eq!(out.normal, Vec3::ZERO);
    }

    #[test]
    fn sweeps_stop_a_slop_short_of_touching() {
        let box_points = cube();
        let a = ShapeProxy {
            points: &box_points,
            radius: 0.0,
        };
        let center = [Vec3::ZERO];
        let ball = ShapeProxy {
            points: &center,
            radius: 0.5,
        };
        let at_rest = Sweep::at_rest(&Transform::IDENTITY);
        let sweep = Sweep {
            local_center: Vec3::ZERO,
            c1: Vec3::new(-5.0, 0.0, 0.0),
            c2: Vec3::new(5.0, 0.0, 0.0),
            q1: Quat::IDENTITY,
            q2: Quat::IDENTITY,
        };
        // The ball's centre stops at x = -(0.5 + 0.5 - slop) = -0.995.
        let hit = time_of_impact(&a, &at_rest, &ball, &sweep, 1.0);
        assert_eq!(hit.state, ToiState::Hit);
        assert!((hit.fraction - 0.4005).abs() < 2e-4, "{hit:?}");
        // Moving away: separated; starting inside: overlapped.
        let away = Sweep {
            c2: Vec3::new(-9.0, 0.0, 0.0),
            ..sweep
        };
        assert_eq!(
            time_of_impact(&a, &at_rest, &ball, &away, 1.0).state,
            ToiState::Separated
        );
        let inside = Sweep {
            c1: Vec3::new(0.2, 0.0, 0.0),
            ..sweep
        };
        assert_eq!(
            time_of_impact(&a, &at_rest, &ball, &inside, 1.0).state,
            ToiState::Overlapped
        );
        // A turning box sweeping past the ball's path catches it with a corner.
        let turning = Sweep {
            local_center: Vec3::ZERO,
            c1: Vec3::new(0.0, 1.2, 0.0),
            c2: Vec3::new(0.0, 1.2, 0.0),
            q1: Quat::IDENTITY,
            q2: Quat::from_rotation_z(0.25 * std::f32::consts::PI),
        };
        let resting_ball = Sweep::at_rest(&Transform::IDENTITY);
        let hit = time_of_impact(&a, &turning, &ball, &resting_ball, 1.0);
        assert_eq!(hit.state, ToiState::Hit, "{hit:?}");
        assert!(0.0 < hit.fraction && hit.fraction < 1.0);
    }

    #[test]
    fn segments_meet_at_their_closest_points() {
        let r = segment_distance(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.25, 1.0, -1.0),
            Vec3::new(0.25, 1.0, 1.0),
        );
        assert!((r.point1 - Vec3::new(0.25, 0.0, 0.0)).length() < 1e-6);
        assert!((r.point2 - Vec3::new(0.25, 1.0, 0.0)).length() < 1e-6);
        let l = line_distance(Vec3::ZERO, Vec3::X, Vec3::new(3.0, 1.0, -1.0), Vec3::Z);
        assert!((l.fraction1 - 3.0).abs() < 1e-6 && !l.is_within_segments());
        assert_eq!(
            closest_point_on_segment(Vec3::ZERO, Vec3::X, Vec3::new(2.0, 1.0, 0.0)),
            Vec3::X
        );
    }
}
