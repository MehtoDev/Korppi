//! Distance between convex shapes (GJK), sweeps and time of impact, ported from Box2D v3's
//! `distance.c`.
//!
//! Shapes enter as [`ShapeProxy2d`]s: up to [`MAX_PROXY_POINTS`] points plus a radius (a circle
//! is one point, a capsule or segment two, a polygon its vertices). Nothing here allocates.

use crate::math::{cross, cross_sv, cross_vs};
use crate::shape::{ChainSegment2d, Shape2d, MAX_POLYGON_VERTICES};
use korppi_math::{Rot2, Transform2d, Vec2};
use serde::{Deserialize, Serialize};

/// Most points a [`ShapeProxy2d`] holds (Box2D `B2_MAX_POLYGON_VERTICES`).
pub const MAX_PROXY_POINTS: usize = MAX_POLYGON_VERTICES;

/// Contact tolerance, in meters (Box2D `B2_LINEAR_SLOP`).
pub const LINEAR_SLOP: f32 = 0.005;

/// A convex shape for the distance and time-of-impact queries: points plus a radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeProxy2d {
    /// The points; only the first `count` are used.
    pub points: [Vec2; MAX_PROXY_POINTS],
    /// Number of points, 1 to [`MAX_PROXY_POINTS`].
    pub count: usize,
    /// Rounding radius around the points.
    pub radius: f32,
}

impl ShapeProxy2d {
    /// A proxy for `points` (at most [`MAX_PROXY_POINTS`] are kept) and `radius`.
    #[must_use]
    pub fn new(points: &[Vec2], radius: f32) -> Self {
        let count = points.len().min(MAX_PROXY_POINTS);
        let mut proxy = Self {
            points: [Vec2::ZERO; MAX_PROXY_POINTS],
            count,
            radius,
        };
        proxy.points[..count].copy_from_slice(&points[..count]);
        proxy
    }

    /// The proxy of a shape in its local frame (Box2D `b2MakeShapeDistanceProxy`); a chain
    /// segment is its segment.
    #[must_use]
    pub fn of_shape(shape: &Shape2d) -> Self {
        match shape {
            Shape2d::Circle(c) => Self::new(&[c.center], c.radius),
            Shape2d::Capsule(c) => Self::new(&[c.point1, c.point2], c.radius),
            Shape2d::Polygon(p) => Self::new(p.vertices(), p.radius),
            Shape2d::Segment(s) | Shape2d::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                Self::new(&[s.point1, s.point2], 0.0)
            }
        }
    }

    fn support(&self, direction: Vec2) -> usize {
        let mut best = 0;
        let mut best_value = self.points[0].dot(direction);
        for i in 1..self.count {
            let value = self.points[i].dot(direction);
            if value > best_value {
                best = i;
                best_value = value;
            }
        }
        best
    }
}

/// Closest points of two segments (Box2D `b2SegmentDistance`, Ericson 5.1.9).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentDistance2d {
    /// Closest point on the first segment.
    pub closest1: Vec2,
    /// Closest point on the second segment.
    pub closest2: Vec2,
    /// Fraction along the first segment.
    pub fraction1: f32,
    /// Fraction along the second segment.
    pub fraction2: f32,
    /// Squared distance between the closest points.
    pub distance_squared: f32,
}

/// Closest points of segments `p1`-`q1` and `p2`-`q2`.
#[must_use]
pub fn segment_distance(p1: Vec2, q1: Vec2, p2: Vec2, q2: Vec2) -> SegmentDistance2d {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let dd1 = d1.dot(d1);
    let dd2 = d2.dot(d2);
    let rd1 = r.dot(d1);
    let rd2 = r.dot(d2);
    let eps_sqr = f32::EPSILON * f32::EPSILON;

    let (f1, f2) = if dd1 < eps_sqr || dd2 < eps_sqr {
        if dd1 >= eps_sqr {
            ((-rd1 / dd1).clamp(0.0, 1.0), 0.0)
        } else if dd2 >= eps_sqr {
            (0.0, (rd2 / dd2).clamp(0.0, 1.0))
        } else {
            (0.0, 0.0)
        }
    } else {
        let d12 = d1.dot(d2);
        let denominator = dd1 * dd2 - d12 * d12;
        let mut f1 = if denominator == 0.0 {
            0.0
        } else {
            ((d12 * rd2 - rd1 * dd2) / denominator).clamp(0.0, 1.0)
        };
        let mut f2 = (d12 * f1 + rd2) / dd2;
        if f2 < 0.0 {
            f2 = 0.0;
            f1 = (-rd1 / dd1).clamp(0.0, 1.0);
        } else if f2 > 1.0 {
            f2 = 1.0;
            f1 = ((d12 - rd1) / dd1).clamp(0.0, 1.0);
        }
        (f1, f2)
    };
    let closest1 = p1 + d1 * f1;
    let closest2 = p2 + d2 * f2;
    SegmentDistance2d {
        closest1,
        closest2,
        fraction1: f1,
        fraction2: f2,
        distance_squared: (closest2 - closest1).length_squared(),
    }
}

/// The support-point indices of a GJK simplex, kept between queries to warm start them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimplexCache2d {
    count: usize,
    index_a: [usize; 3],
    index_b: [usize; 3],
}

impl SimplexCache2d {
    /// An empty cache: the next query starts from scratch.
    pub const EMPTY: Self = Self {
        count: 0,
        index_a: [0; 3],
        index_b: [0; 3],
    };

    /// Number of vertices the last query's simplex had (Box2D `b2SimplexCache::count`).
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Vertex `i` of the last simplex on proxy A.
    #[must_use]
    pub const fn index_a(&self, i: usize) -> usize {
        self.index_a[i]
    }

    /// Vertex `i` of the last simplex on proxy B.
    #[must_use]
    pub const fn index_b(&self, i: usize) -> usize {
        self.index_b[i]
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SimplexVertex {
    w_a: Vec2,
    w_b: Vec2,
    w: Vec2,
    a: f32,
    index_a: usize,
    index_b: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct Simplex {
    v: [SimplexVertex; 3],
    count: usize,
}

impl Simplex {
    fn from_cache(cache: &SimplexCache2d, a: &ShapeProxy2d, b: &ShapeProxy2d) -> Self {
        let mut s = Self {
            count: cache.count,
            ..Self::default()
        };
        for i in 0..s.count {
            let (ia, ib) = (cache.index_a[i], cache.index_b[i]);
            s.v[i] = SimplexVertex {
                w_a: a.points[ia],
                w_b: b.points[ib],
                w: a.points[ia] - b.points[ib],
                a: -1.0,
                index_a: ia,
                index_b: ib,
            };
        }
        if s.count == 0 {
            s.v[0] = SimplexVertex {
                w_a: a.points[0],
                w_b: b.points[0],
                w: a.points[0] - b.points[0],
                a: 1.0,
                index_a: 0,
                index_b: 0,
            };
            s.count = 1;
        }
        s
    }

    fn cache(&self) -> SimplexCache2d {
        let mut cache = SimplexCache2d {
            count: self.count,
            ..SimplexCache2d::default()
        };
        for i in 0..self.count {
            cache.index_a[i] = self.v[i].index_a;
            cache.index_b[i] = self.v[i].index_b;
        }
        cache
    }

    fn witness_points(&self) -> (Vec2, Vec2) {
        let v = &self.v;
        match self.count {
            1 => (v[0].w_a, v[0].w_b),
            2 => (
                v[0].w_a * v[0].a + v[1].w_a * v[1].a,
                v[0].w_b * v[0].a + v[1].w_b * v[1].a,
            ),
            3 => {
                let a = v[0].w_a * v[0].a + v[1].w_a * v[1].a + v[2].w_a * v[2].a;
                (a, a)
            }
            _ => (Vec2::ZERO, Vec2::ZERO),
        }
    }

    /// Reduces a segment simplex to the part closest to the origin; returns the search
    /// direction toward the origin.
    fn solve2(&mut self) -> Vec2 {
        let (w1, w2) = (self.v[0].w, self.v[1].w);
        let e12 = w2 - w1;
        let d12_2 = -w1.dot(e12);
        if d12_2 <= 0.0 {
            self.v[0].a = 1.0;
            self.count = 1;
            return -w1;
        }
        let d12_1 = w2.dot(e12);
        if d12_1 <= 0.0 {
            self.v[1].a = 1.0;
            self.count = 1;
            self.v[0] = self.v[1];
            return -w2;
        }
        let inv_d12 = 1.0 / (d12_1 + d12_2);
        self.v[0].a = d12_1 * inv_d12;
        self.v[1].a = d12_2 * inv_d12;
        self.count = 2;
        cross_sv(cross(w1 + w2, e12), e12)
    }

    /// Reduces a triangle simplex to the part closest to the origin; returns the search
    /// direction, zero when the triangle contains the origin.
    fn solve3(&mut self) -> Vec2 {
        let (w1, w2, w3) = (self.v[0].w, self.v[1].w, self.v[2].w);
        let e12 = w2 - w1;
        let d12_1 = w2.dot(e12);
        let d12_2 = -w1.dot(e12);
        let e13 = w3 - w1;
        let d13_1 = w3.dot(e13);
        let d13_2 = -w1.dot(e13);
        let e23 = w3 - w2;
        let d23_1 = w3.dot(e23);
        let d23_2 = -w2.dot(e23);
        let n123 = cross(e12, e13);
        let d123_1 = n123 * cross(w2, w3);
        let d123_2 = n123 * cross(w3, w1);
        let d123_3 = n123 * cross(w1, w2);

        if d12_2 <= 0.0 && d13_2 <= 0.0 {
            self.v[0].a = 1.0;
            self.count = 1;
            return -w1;
        }
        if d12_1 > 0.0 && d12_2 > 0.0 && d123_3 <= 0.0 {
            let inv = 1.0 / (d12_1 + d12_2);
            self.v[0].a = d12_1 * inv;
            self.v[1].a = d12_2 * inv;
            self.count = 2;
            return cross_sv(cross(w1 + w2, e12), e12);
        }
        if d13_1 > 0.0 && d13_2 > 0.0 && d123_2 <= 0.0 {
            let inv = 1.0 / (d13_1 + d13_2);
            self.v[0].a = d13_1 * inv;
            self.v[2].a = d13_2 * inv;
            self.count = 2;
            self.v[1] = self.v[2];
            return cross_sv(cross(w1 + w3, e13), e13);
        }
        if d12_1 <= 0.0 && d23_2 <= 0.0 {
            self.v[1].a = 1.0;
            self.count = 1;
            self.v[0] = self.v[1];
            return -w2;
        }
        if d13_1 <= 0.0 && d23_1 <= 0.0 {
            self.v[2].a = 1.0;
            self.count = 1;
            self.v[0] = self.v[2];
            return -w3;
        }
        if d23_1 > 0.0 && d23_2 > 0.0 && d123_1 <= 0.0 {
            let inv = 1.0 / (d23_1 + d23_2);
            self.v[1].a = d23_1 * inv;
            self.v[2].a = d23_2 * inv;
            self.count = 2;
            self.v[0] = self.v[2];
            return cross_sv(cross(w2 + w3, e23), e23);
        }
        let inv = 1.0 / (d123_1 + d123_2 + d123_3);
        self.v[0].a = d123_1 * inv;
        self.v[1].a = d123_2 * inv;
        self.v[2].a = d123_3 * inv;
        self.count = 3;
        Vec2::ZERO
    }
}

/// Result of [`shape_distance`], in proxy A's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceOutput2d {
    /// Closest point on A.
    pub point_a: Vec2,
    /// Closest point on B.
    pub point_b: Vec2,
    /// Unit direction from A toward B (zero when overlapping).
    pub normal: Vec2,
    /// Distance between the shapes, `0` when they overlap.
    pub distance: f32,
}

/// Distance between proxy A and proxy B placed by `transform_b` in A's frame (Box2D
/// `b2ShapeDistance`, GJK). `cache` warm starts the query and is updated. With `use_radii`, the
/// radii are subtracted and the closest points moved onto the rounded surfaces.
#[must_use]
pub fn shape_distance(
    proxy_a: &ShapeProxy2d,
    proxy_b: &ShapeProxy2d,
    transform_b: &Transform2d,
    cache: &mut SimplexCache2d,
    use_radii: bool,
) -> DistanceOutput2d {
    let mut local_b = *proxy_b;
    for point in &mut local_b.points[..local_b.count] {
        *point = transform_b.transform_point(*point);
    }

    let mut simplex = Simplex::from_cache(cache, proxy_a, &local_b);
    let mut non_unit_normal = Vec2::ZERO;
    let mut save_a = [0usize; 3];
    let mut save_b = [0usize; 3];

    for _ in 0..20 {
        let save_count = simplex.count;
        for i in 0..save_count {
            save_a[i] = simplex.v[i].index_a;
            save_b[i] = simplex.v[i].index_b;
        }

        let d = match simplex.count {
            1 => -simplex.v[0].w,
            2 => simplex.solve2(),
            _ => simplex.solve3(),
        };

        if simplex.count == 3 || d.dot(d) < 1000.0 * f32::MIN_POSITIVE {
            // The origin is inside the simplex: overlap.
            *cache = simplex.cache();
            let (point_a, point_b) = simplex.witness_points();
            return DistanceOutput2d {
                point_a,
                point_b,
                normal: Vec2::ZERO,
                distance: 0.0,
            };
        }

        non_unit_normal = d;
        let index_a = proxy_a.support(d);
        let index_b = local_b.support(-d);
        let duplicate = (0..save_count).any(|i| save_a[i] == index_a && save_b[i] == index_b);
        if duplicate {
            break;
        }
        let w_a = proxy_a.points[index_a];
        let w_b = local_b.points[index_b];
        simplex.v[simplex.count] = SimplexVertex {
            w_a,
            w_b,
            w: w_a - w_b,
            a: 0.0,
            index_a,
            index_b,
        };
        simplex.count += 1;
    }

    let normal = non_unit_normal.normalize_or_zero();
    let (mut point_a, mut point_b) = simplex.witness_points();
    let mut distance = (point_b - point_a).length();
    *cache = simplex.cache();
    if use_radii {
        distance = (distance - proxy_a.radius - proxy_b.radius).max(0.0);
        point_a += normal * proxy_a.radius;
        point_b -= normal * proxy_b.radius;
    }
    DistanceOutput2d {
        point_a,
        point_b,
        normal,
        distance,
    }
}

/// Motion of a body over a step: its centre of mass moves linearly from `c1` to `c2` and its
/// rotation from `q1` to `q2` (Box2D `b2Sweep`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sweep2d {
    /// Centre of mass in body-local coordinates.
    pub local_center: Vec2,
    /// World centre of mass at the start.
    pub c1: Vec2,
    /// World centre of mass at the end.
    pub c2: Vec2,
    /// Rotation at the start.
    pub q1: Rot2,
    /// Rotation at the end.
    pub q2: Rot2,
}

impl Sweep2d {
    /// A body that does not move, at `transform`, with centre of mass `local_center`.
    #[must_use]
    pub fn still(transform: &Transform2d, local_center: Vec2) -> Self {
        let c = transform.transform_point(local_center);
        Self {
            local_center,
            c1: c,
            c2: c,
            q1: transform.rotation,
            q2: transform.rotation,
        }
    }

    /// The body transform at `time` in [0, 1] (Box2D `b2GetSweepTransform`): centre and
    /// rotation interpolated linearly, the rotation renormalized.
    #[must_use]
    pub fn transform(&self, time: f32) -> Transform2d {
        let center = self.c1 * (1.0 - time) + self.c2 * time;
        let rotation = Rot2::new(
            (1.0 - time) * self.q1.c + time * self.q2.c,
            (1.0 - time) * self.q1.s + time * self.q2.s,
        )
        .normalize();
        Transform2d::from_translation_rotation(
            center - rotation.rotate(self.local_center),
            rotation,
        )
    }
}

/// How a time-of-impact query ended (Box2D `b2TOIState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToiState2d {
    /// The root finder gave up; `fraction` is the last safe time.
    Failed,
    /// The shapes overlap at the start.
    Overlapped,
    /// The shapes touch at `fraction`.
    Hit,
    /// The shapes stay apart up to the maximum fraction.
    Separated,
}

/// Result of [`time_of_impact`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToiOutput2d {
    /// How the query ended.
    pub state: ToiState2d,
    /// Sweep time of the impact, or of the last safe time.
    pub fraction: f32,
    /// World contact point at a hit.
    pub point: Vec2,
    /// World normal from A toward B at a hit.
    pub normal: Vec2,
}

/// Inputs to [`time_of_impact`] (Box2D `b2TOIInput`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToiInput2d {
    /// Shape A in its body frame.
    pub proxy_a: ShapeProxy2d,
    /// Shape B in its body frame.
    pub proxy_b: ShapeProxy2d,
    /// Motion of body A.
    pub sweep_a: Sweep2d,
    /// Motion of body B.
    pub sweep_b: Sweep2d,
    /// Sweep time not to look past, in [0, 1].
    pub max_fraction: f32,
    /// The world's linear slop ([`LINEAR_SLOP`] in metres): the sweep stops this close to the
    /// impact.
    pub linear_slop: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SeparationKind {
    Points,
    FaceA,
    FaceB,
}

/// A separating axis between the two sweeps (Box2D `b2SeparationFunction`).
struct SeparationFunction<'a> {
    input: &'a ToiInput2d,
    local_point: Vec2,
    axis: Vec2,
    kind: SeparationKind,
}

impl<'a> SeparationFunction<'a> {
    fn new(cache: &SimplexCache2d, input: &'a ToiInput2d, t1: f32) -> Self {
        let (a, b) = (&input.proxy_a, &input.proxy_b);
        let xf_a = input.sweep_a.transform(t1);
        let xf_b = input.sweep_b.transform(t1);
        if cache.count == 1 {
            let point_a = xf_a.transform_point(a.points[cache.index_a[0]]);
            let point_b = xf_b.transform_point(b.points[cache.index_b[0]]);
            return Self {
                input,
                local_point: Vec2::ZERO,
                axis: (point_b - point_a).normalize_or_zero(),
                kind: SeparationKind::Points,
            };
        }
        if cache.index_a[0] == cache.index_a[1] {
            // Two points on B, one on A.
            let b1 = b.points[cache.index_b[0]];
            let b2 = b.points[cache.index_b[1]];
            let mut axis = cross_vs(b2 - b1, 1.0).normalize_or_zero();
            let normal = xf_b.rotation.rotate(axis);
            let local_point = (b1 + b2) * 0.5;
            let point_b = xf_b.transform_point(local_point);
            let point_a = xf_a.transform_point(a.points[cache.index_a[0]]);
            if (point_a - point_b).dot(normal) < 0.0 {
                axis = -axis;
            }
            return Self {
                input,
                local_point,
                axis,
                kind: SeparationKind::FaceB,
            };
        }
        // Two points on A, one or two on B.
        let a1 = a.points[cache.index_a[0]];
        let a2 = a.points[cache.index_a[1]];
        let mut axis = cross_vs(a2 - a1, 1.0).normalize_or_zero();
        let normal = xf_a.rotation.rotate(axis);
        let local_point = (a1 + a2) * 0.5;
        let point_a = xf_a.transform_point(local_point);
        let point_b = xf_b.transform_point(b.points[cache.index_b[0]]);
        if (point_b - point_a).dot(normal) < 0.0 {
            axis = -axis;
        }
        Self {
            input,
            local_point,
            axis,
            kind: SeparationKind::FaceA,
        }
    }

    /// The deepest points along the axis at time `t` and their separation.
    fn find_min_separation(&self, t: f32) -> (usize, usize, f32) {
        let (a, b) = (&self.input.proxy_a, &self.input.proxy_b);
        let xf_a = self.input.sweep_a.transform(t);
        let xf_b = self.input.sweep_b.transform(t);
        match self.kind {
            SeparationKind::Points => {
                let index_a = a.support(xf_a.rotation.inv_rotate(self.axis));
                let index_b = b.support(xf_b.rotation.inv_rotate(-self.axis));
                let point_a = xf_a.transform_point(a.points[index_a]);
                let point_b = xf_b.transform_point(b.points[index_b]);
                (index_a, index_b, (point_b - point_a).dot(self.axis))
            }
            SeparationKind::FaceA => {
                let normal = xf_a.rotation.rotate(self.axis);
                let point_a = xf_a.transform_point(self.local_point);
                let index_b = b.support(xf_b.rotation.inv_rotate(-normal));
                let point_b = xf_b.transform_point(b.points[index_b]);
                (0, index_b, (point_b - point_a).dot(normal))
            }
            SeparationKind::FaceB => {
                let normal = xf_b.rotation.rotate(self.axis);
                let point_b = xf_b.transform_point(self.local_point);
                let index_a = a.support(xf_a.rotation.inv_rotate(-normal));
                let point_a = xf_a.transform_point(a.points[index_a]);
                (index_a, 0, (point_a - point_b).dot(normal))
            }
        }
    }

    /// Separation of the given points along the axis at time `t`.
    fn evaluate(&self, index_a: usize, index_b: usize, t: f32) -> f32 {
        let (a, b) = (&self.input.proxy_a, &self.input.proxy_b);
        let xf_a = self.input.sweep_a.transform(t);
        let xf_b = self.input.sweep_b.transform(t);
        match self.kind {
            SeparationKind::Points => {
                let point_a = xf_a.transform_point(a.points[index_a]);
                let point_b = xf_b.transform_point(b.points[index_b]);
                (point_b - point_a).dot(self.axis)
            }
            SeparationKind::FaceA => {
                let normal = xf_a.rotation.rotate(self.axis);
                let point_a = xf_a.transform_point(self.local_point);
                let point_b = xf_b.transform_point(b.points[index_b]);
                (point_b - point_a).dot(normal)
            }
            SeparationKind::FaceB => {
                let normal = xf_b.rotation.rotate(self.axis);
                let point_b = xf_b.transform_point(self.local_point);
                let point_a = xf_a.transform_point(a.points[index_a]);
                (point_a - point_b).dot(normal)
            }
        }
    }
}

/// Time of impact of two moving shapes (Box2D `b2TimeOfImpact`): conservative advancement
/// along local separating axes. The shapes are considered touching at a distance of
/// `radius_a + radius_b - LINEAR_SLOP` (at least `LINEAR_SLOP`), within a quarter slop.
#[must_use]
pub fn time_of_impact(input: &ToiInput2d) -> ToiOutput2d {
    let mut output = ToiOutput2d {
        state: ToiState2d::Failed,
        fraction: input.max_fraction,
        point: Vec2::ZERO,
        normal: Vec2::ZERO,
    };
    let (proxy_a, proxy_b) = (&input.proxy_a, &input.proxy_b);
    let t_max = input.max_fraction;
    let total_radius = proxy_a.radius + proxy_b.radius;
    let slop = input.linear_slop;
    let target = slop.max(total_radius - slop);
    let tolerance = 0.25 * slop;

    let mut t1 = 0.0f32;
    let mut cache = SimplexCache2d::default();
    let hit_point = |point_a: Vec2, point_b: Vec2, normal: Vec2| {
        let p_a = point_a + normal * proxy_a.radius;
        let p_b = point_b - normal * proxy_b.radius;
        (p_a + p_b) * 0.5
    };

    for distance_iterations in 1..=20 {
        let xf_a = input.sweep_a.transform(t1);
        let xf_b = input.sweep_b.transform(t1);
        let relative = xf_a.inv_mul(&xf_b);
        let distance = shape_distance(proxy_a, proxy_b, &relative, &mut cache, false);
        let world_normal = xf_a.rotation.rotate(distance.normal);
        let world_a = xf_a.transform_point(distance.point_a);
        let world_b = xf_a.transform_point(distance.point_b);

        if distance.distance <= 0.0 {
            output.state = ToiState2d::Overlapped;
            output.fraction = 0.0;
            return output;
        }
        if distance.distance <= target + tolerance {
            output.state = ToiState2d::Hit;
            output.point = hit_point(world_a, world_b, world_normal);
            output.normal = world_normal;
            output.fraction = t1;
            return output;
        }

        let function = SeparationFunction::new(&cache, input, t1);
        let mut t2 = t_max;
        for _push_back in 0..MAX_PROXY_POINTS {
            let (index_a, index_b, mut s2) = function.find_min_separation(t2);
            if s2 > target + tolerance {
                output.state = ToiState2d::Separated;
                output.fraction = t_max;
                return output;
            }
            if s2 > target - tolerance {
                t1 = t2;
                break;
            }
            let mut s1 = function.evaluate(index_a, index_b, t1);
            if s1 < target - tolerance {
                output.state = ToiState2d::Failed;
                output.fraction = t1;
                return output;
            }
            if s1 <= target + tolerance {
                output.state = ToiState2d::Hit;
                output.point = hit_point(world_a, world_b, world_normal);
                output.normal = world_normal;
                output.fraction = t1;
                return output;
            }

            // Root of separation(t) = target: alternate bisection and false position.
            let (mut a1, mut a2) = (t1, t2);
            for root_iteration in 0..50 {
                let t = if root_iteration & 1 == 1 {
                    a1 + (target - s1) * (a2 - a1) / (s2 - s1)
                } else {
                    0.5 * (a1 + a2)
                };
                let s = function.evaluate(index_a, index_b, t);
                if (s - target).abs() < tolerance {
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
            }
        }

        if distance_iterations == 20 {
            output.state = ToiState2d::Failed;
            output.point = hit_point(world_a, world_b, world_normal);
            output.normal = world_normal;
            output.fraction = t1;
        }
    }
    output
}

/// A ray: from `origin` along `translation`, up to `max_fraction` of it (Box2D
/// `b2RayCastInput`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayCastInput2d {
    /// Start point.
    pub origin: Vec2,
    /// The ray's whole length and direction.
    pub translation: Vec2,
    /// The fraction of `translation` to consider, usually 1.
    pub max_fraction: f32,
}

/// A shape cast: `proxy` moved by `translation`, up to `max_fraction` of it (Box2D
/// `b2ShapeCastInput`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeCastInput2d {
    /// The cast shape.
    pub proxy: ShapeProxy2d,
    /// The cast's whole length and direction.
    pub translation: Vec2,
    /// The fraction of `translation` to consider, usually 1.
    pub max_fraction: f32,
    /// Whether a rounded shape already touching may move a little closer before it counts as a
    /// hit, as character movers do.
    pub can_encroach: bool,
}

/// The result of a ray or shape cast (Box2D `b2CastOutput`), in the frame of the input. An
/// initial overlap is a hit at fraction 0 without a normal.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CastOutput2d {
    /// Surface normal at the hit point.
    pub normal: Vec2,
    /// Hit point.
    pub point: Vec2,
    /// Fraction of the translation at the hit.
    pub fraction: f32,
    /// Iterations the cast took.
    pub iterations: u32,
    /// Whether the cast hit.
    pub hit: bool,
}

/// A shape cast of proxy B, placed in A's frame by `transform` and moving by `translation_b`,
/// against proxy A (Box2D `b2ShapeCastPairInput`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeCastPairInput2d {
    /// The fixed shape.
    pub proxy_a: ShapeProxy2d,
    /// The moving shape.
    pub proxy_b: ShapeProxy2d,
    /// B's pose in A's frame.
    pub transform: Transform2d,
    /// B's translation in A's frame.
    pub translation_b: Vec2,
    /// The fraction of the translation to consider, usually 1.
    pub max_fraction: f32,
    /// See [`ShapeCastInput2d::can_encroach`].
    pub can_encroach: bool,
}

/// Linear shape cast of proxy B against fixed proxy A (Box2D `b2ShapeCast`): conservative
/// advancement on GJK distance, in A's frame. B touching A at the start is a hit at fraction 0
/// (unless it may encroach), at a point midway between the surfaces and without a normal.
#[must_use]
pub fn shape_cast(input: &ShapeCastPairInput2d) -> CastOutput2d {
    let total_radius = input.proxy_a.radius + input.proxy_b.radius;
    let mut target = LINEAR_SLOP.max(total_radius - LINEAR_SLOP);
    let tolerance = 0.25 * LINEAR_SLOP;
    let mut cache = SimplexCache2d::EMPTY;
    let mut fraction = 0.0f32;
    let mut transform = input.transform;
    let delta2 = input.translation_b;
    let mut output = CastOutput2d::default();
    for iteration in 0..20 {
        output.iterations += 1;
        let distance = shape_distance(
            &input.proxy_a,
            &input.proxy_b,
            &transform,
            &mut cache,
            false,
        );
        if distance.distance < target + tolerance {
            if iteration == 0 {
                if input.can_encroach && distance.distance > 2.0 * LINEAR_SLOP {
                    target = distance.distance - LINEAR_SLOP;
                } else {
                    // Initial overlap: a common point.
                    output.hit = true;
                    let c1 = distance.point_a + distance.normal * input.proxy_a.radius;
                    let c2 = distance.point_b + distance.normal * -input.proxy_b.radius;
                    output.point = crate::math::lerp(c1, c2, 0.5);
                    return output;
                }
            } else {
                output.fraction = fraction;
                output.point = distance.point_a + distance.normal * input.proxy_a.radius;
                output.normal = distance.normal;
                output.hit = true;
                return output;
            }
        }
        // Are the shapes approaching?
        let denominator = delta2.dot(distance.normal);
        if denominator >= 0.0 {
            return output;
        }
        fraction += (target - distance.distance) / denominator;
        if fraction >= input.max_fraction {
            return output;
        }
        transform.translation = input.transform.translation + delta2 * fraction;
    }
    output
}
