//! Ray casts, shape casts, point tests and mover collision of shapes in their local frames,
//! ported from Box2D v3.2's `geometry.c` and `shape.c`.
//!
//! Casts take a translation and report the fraction of it at the hit ([`CastOutput2d`]); a
//! cast that starts inside a shape is a hit at fraction 0 without a normal. Segments have no
//! inside, and chain segments are hit from their front only.

use crate::distance::{
    shape_cast, shape_distance, CastOutput2d, RayCastInput2d, ShapeCastInput2d,
    ShapeCastPairInput2d, ShapeProxy2d, SimplexCache2d,
};
use crate::math::{
    clamp, cross, distance_squared, length_and_normalize, lerp, normalize, right_perp,
};
use crate::mover::{Plane2d, PlaneResult2d};
use crate::shape::{Capsule2d, ChainSegment2d, Circle2d, Polygon2d, Segment2d, Shape2d};
use korppi_math::{Transform2d, Vec2};

impl Circle2d {
    /// Whether `point` is inside the circle (Box2D `b2PointInCircle`).
    #[must_use]
    pub fn test_point(&self, point: Vec2) -> bool {
        distance_squared(point, self.center) <= self.radius * self.radius
    }

    /// Ray cast (Box2D `b2RayCastCircle`, after "Precision Improvements for Ray / Sphere
    /// Intersection", Ray Tracing Gems 2019).
    #[must_use]
    pub fn ray_cast(&self, input: &RayCastInput2d) -> CastOutput2d {
        let p = self.center;
        let mut output = CastOutput2d::default();
        // Shift the ray so that the circle's centre is the origin.
        let s = input.origin - p;
        let r = self.radius;
        let rr = r * r;
        let (length, d) = length_and_normalize(input.translation);
        let initial_overlap = |output: &mut CastOutput2d| {
            if s.length_squared() < rr {
                output.point = input.origin;
                output.hit = true;
            }
        };
        if length == 0.0 {
            initial_overlap(&mut output);
            return output;
        }
        // The point of the line closest to the centre.
        let t = -s.dot(d);
        let c = s + d * t;
        let cc = c.dot(c);
        if cc > rr {
            return output;
        }
        let h = (rr - cc).sqrt();
        let fraction = t - h;
        if fraction < 0.0 || input.max_fraction * length < fraction {
            initial_overlap(&mut output);
            return output;
        }
        let hit = s + d * fraction;
        output.fraction = fraction / length;
        output.normal = normalize(hit);
        output.point = p + output.normal * self.radius;
        output.hit = true;
        output
    }
}

impl Capsule2d {
    /// Whether `point` is inside the capsule (Box2D `b2PointInCapsule`).
    #[must_use]
    pub fn test_point(&self, point: Vec2) -> bool {
        let rr = self.radius * self.radius;
        let (p1, p2) = (self.point1, self.point2);
        let d = p2 - p1;
        let dd = d.dot(d);
        if dd == 0.0 {
            return distance_squared(point, p1) <= rr;
        }
        let t = clamp((point - p1).dot(d) / dd, 0.0, 1.0);
        let c = p1 + d * t;
        distance_squared(point, c) <= rr
    }

    /// Ray cast (Box2D `b2RayCastCapsule`).
    #[must_use]
    pub fn ray_cast(&self, input: &RayCastInput2d) -> CastOutput2d {
        let mut output = CastOutput2d::default();
        let (v1, v2) = (self.point1, self.point2);
        let (capsule_length, a) = length_and_normalize(v2 - v1);
        let circle = |center| Circle2d::new(center, self.radius).ray_cast(input);
        if capsule_length < f32::EPSILON {
            return circle(v1);
        }
        let p1 = input.origin;
        let d = input.translation;
        // From the capsule's start to the ray's.
        let q = p1 - v1;
        let qa = q.dot(a);
        // The part of q perpendicular to the axis.
        let qp = q + a * -qa;
        let radius = self.radius;
        // Does the ray start inside the infinite capsule?
        if qp.dot(qp) < radius * radius {
            if qa < 0.0 {
                return circle(v1);
            }
            if qa > capsule_length {
                return circle(v2);
            }
            // Inside the capsule.
            output.point = input.origin;
            output.hit = true;
            return output;
        }
        // Perpendicular to the axis, pointing right.
        let mut n = Vec2::new(a.y, -a.x);
        let (ray_length, u) = length_and_normalize(d);
        // Intersect the ray with the infinite capsule by Cramer's rule.
        let den = -a.x * u.y + u.x * a.y;
        if -f32::EPSILON < den && den < f32::EPSILON {
            // Parallel to the axis, outside the infinite capsule.
            return output;
        }
        let b1 = q - n * radius;
        let b2 = q + n * radius;
        let inv_den = 1.0 / den;
        let s21 = (a.x * b1.y - b1.x * a.y) * inv_den;
        let s22 = (a.x * b2.y - b2.x * a.y) * inv_den;
        let (s2, b) = if s21 < s22 {
            (s21, b1)
        } else {
            n = -n;
            (s22, b2)
        };
        if s2 < 0.0 || input.max_fraction * ray_length < s2 {
            return output;
        }
        let s1 = (-b.x * u.y + u.x * b.y) * inv_den;
        if s1 < 0.0 {
            // Behind the segment.
            circle(v1)
        } else if capsule_length < s1 {
            // Ahead of the segment.
            circle(v2)
        } else {
            output.fraction = s2 / ray_length;
            output.point = lerp(v1, v2, s1 / capsule_length) + n * self.radius;
            output.normal = n;
            output.hit = true;
            output
        }
    }
}

impl Polygon2d {
    /// Whether `point` is inside the polygon, its rounding included (Box2D
    /// `b2PointInPolygon`).
    #[must_use]
    pub fn test_point(&self, point: Vec2) -> bool {
        let output = shape_distance(
            &ShapeProxy2d::new(self.vertices(), 0.0),
            &ShapeProxy2d::new(&[point], 0.0),
            &Transform2d::IDENTITY,
            &mut SimplexCache2d::default(),
            false,
        );
        output.distance <= self.radius
    }

    /// Ray cast (Box2D `b2RayCastPolygon`); a rounded polygon is shape cast with a point.
    #[must_use]
    pub fn ray_cast(&self, input: &RayCastInput2d) -> CastOutput2d {
        if self.radius != 0.0 {
            return shape_cast(&ShapeCastPairInput2d {
                proxy_a: ShapeProxy2d::new(self.vertices(), self.radius),
                proxy_b: ShapeProxy2d::new(&[input.origin], 0.0),
                transform: Transform2d::IDENTITY,
                translation_b: input.translation,
                max_fraction: input.max_fraction,
                can_encroach: false,
            });
        }
        // Work relative to the first vertex, as the polygon may be far from the origin.
        let base = self.vertices()[0];
        let p1 = input.origin - base;
        let d = input.translation;
        let (mut lower, mut upper) = (0.0f32, input.max_fraction);
        let mut index = None;
        let mut output = CastOutput2d::default();
        for (edge, (&vertex, &normal)) in self.vertices().iter().zip(self.normals()).enumerate() {
            // p = p1 + a * d, normal · (p - v) = 0.
            let numerator = normal.dot((vertex - base) - p1);
            let denominator = normal.dot(d);
            if denominator == 0.0 {
                // Parallel, and outside this edge.
                if numerator < 0.0 {
                    return output;
                }
            } else if denominator < 0.0 && numerator < lower * denominator {
                // The ray enters this half-space.
                lower = numerator / denominator;
                index = Some(edge);
            } else if denominator > 0.0 && numerator < upper * denominator {
                // The ray leaves this half-space.
                upper = numerator / denominator;
            }
            if upper < lower {
                return output;
            }
        }
        if let Some(edge) = index {
            output.fraction = lower;
            output.normal = self.normals()[edge];
            output.point = input.origin + d * lower;
        } else {
            // Inside the polygon.
            output.point = input.origin;
        }
        output.hit = true;
        output
    }
}

impl Segment2d {
    /// Ray cast from either side (Box2D `b2RayCastSegment`).
    #[must_use]
    pub fn ray_cast(&self, input: &RayCastInput2d) -> CastOutput2d {
        ray_cast_segment(self, input, false)
    }
}

impl ChainSegment2d {
    /// Ray cast from the front only (Box2D `b2RayCastSegment` with `oneSided`): a ray
    /// starting behind the segment misses.
    #[must_use]
    pub fn ray_cast(&self, input: &RayCastInput2d) -> CastOutput2d {
        ray_cast_segment(&self.segment, input, true)
    }
}

/// Box2D `b2RayCastSegment`.
fn ray_cast_segment(segment: &Segment2d, input: &RayCastInput2d, one_sided: bool) -> CastOutput2d {
    let mut output = CastOutput2d::default();
    let (v1, v2) = (segment.point1, segment.point2);
    if one_sided && cross(input.origin - v1, v2 - v1) < 0.0 {
        // The ray starts on the left side.
        return output;
    }
    let p1 = input.origin;
    let d = input.translation;
    let (length, e) = length_and_normalize(v2 - v1);
    if length == 0.0 {
        return output;
    }
    // The normal points to the right, looking from v1 towards v2.
    let mut normal = right_perp(e);
    // Intersect the ray with the segment's line.
    let numerator = normal.dot(v1 - p1);
    let denominator = normal.dot(d);
    if denominator == 0.0 {
        return output;
    }
    let t = numerator / denominator;
    if t < 0.0 || input.max_fraction < t {
        return output;
    }
    let p = p1 + d * t;
    let s = (p - v1).dot(e);
    if s < 0.0 || length < s {
        return output;
    }
    if numerator > 0.0 {
        normal = -normal;
    }
    output.fraction = t;
    output.point = p;
    output.normal = normal;
    output.hit = true;
    output
}

/// Box2D `b2CollideMover*`: the plane between a capsule mover and a shape of core proxy `core`
/// rounded by `radius`, if they touch.
fn collide_mover(mover: &Capsule2d, core: &ShapeProxy2d, radius: f32) -> Option<PlaneResult2d> {
    let output = shape_distance(
        core,
        &ShapeProxy2d::new(&[mover.point1, mover.point2], mover.radius),
        &Transform2d::IDENTITY,
        &mut SimplexCache2d::default(),
        false,
    );
    let total_radius = mover.radius + radius;
    (output.distance <= total_radius).then_some(PlaneResult2d {
        plane: Plane2d::new(output.normal, total_radius - output.distance),
        point: output.point_a,
    })
}

impl Shape2d {
    /// Whether `point` (in the shape's frame) is inside the shape (Box2D
    /// `b2Shape_TestPoint`); segments and chain segments have no inside.
    #[must_use]
    pub fn test_point(&self, point: Vec2) -> bool {
        match self {
            Self::Circle(c) => c.test_point(point),
            Self::Capsule(c) => c.test_point(point),
            Self::Polygon(p) => p.test_point(point),
            Self::Segment(_) | Self::ChainSegment(_) => false,
        }
    }

    /// Ray cast in the shape's frame (Box2D `b2RayCastShape`).
    #[must_use]
    pub fn ray_cast(&self, input: &RayCastInput2d) -> CastOutput2d {
        match self {
            Self::Circle(c) => c.ray_cast(input),
            Self::Capsule(c) => c.ray_cast(input),
            Self::Polygon(p) => p.ray_cast(input),
            Self::Segment(s) => s.ray_cast(input),
            Self::ChainSegment(c) => c.ray_cast(input),
        }
    }

    /// Shape cast in the shape's frame (Box2D `b2ShapeCastShape`): `input.proxy` moves, this
    /// shape stays. A chain segment ignores casts that start behind it.
    #[must_use]
    pub fn shape_cast(&self, input: &ShapeCastInput2d) -> CastOutput2d {
        if input.proxy.count == 0 {
            return CastOutput2d::default();
        }
        if let Self::ChainSegment(chain) = self {
            let points = &input.proxy.points[..input.proxy.count];
            let sum = points[1..].iter().fold(points[0], |sum, &p| sum + p);
            let centroid = sum * (1.0 / input.proxy.count as f32);
            let (p1, p2) = (chain.segment.point1, chain.segment.point2);
            if cross(centroid - p1, p2 - p1) < 0.0 {
                // The cast starts behind the segment.
                return CastOutput2d::default();
            }
        }
        let (points, radius): (&[Vec2], f32) = match self {
            Self::Circle(c) => (std::slice::from_ref(&c.center), c.radius),
            Self::Capsule(c) => (&[c.point1, c.point2], c.radius),
            Self::Polygon(p) => (p.vertices(), p.radius),
            Self::Segment(s) | Self::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                (&[s.point1, s.point2], 0.0)
            }
        };
        shape_cast(&ShapeCastPairInput2d {
            proxy_a: ShapeProxy2d::new(points, radius),
            proxy_b: input.proxy,
            transform: Transform2d::IDENTITY,
            translation_b: input.translation,
            max_fraction: input.max_fraction,
            can_encroach: input.can_encroach,
        })
    }

    /// The plane between a capsule mover (in the shape's frame) and the shape, if they touch
    /// (Box2D `b2CollideMover`).
    #[must_use]
    pub fn collide_mover(&self, mover: &Capsule2d) -> Option<PlaneResult2d> {
        match self {
            Self::Circle(c) => collide_mover(mover, &ShapeProxy2d::new(&[c.center], 0.0), c.radius),
            Self::Capsule(c) => collide_mover(
                mover,
                &ShapeProxy2d::new(&[c.point1, c.point2], 0.0),
                c.radius,
            ),
            Self::Polygon(p) => {
                collide_mover(mover, &ShapeProxy2d::new(p.vertices(), p.radius), p.radius)
            }
            Self::Segment(s) | Self::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                collide_mover(mover, &ShapeProxy2d::new(&[s.point1, s.point2], 0.0), 0.0)
            }
        }
    }

    /// The shape's extent across `line` (Box2D `b2GetShapeProjectedPerimeter`), which an
    /// explosion pushes on.
    #[must_use]
    pub fn projected_perimeter(&self, line: Vec2) -> f32 {
        match self {
            Self::Capsule(c) => (c.point2 - c.point1).dot(line).abs() + 2.0 * c.radius,
            Self::Circle(c) => 2.0 * c.radius,
            Self::Polygon(p) => {
                let points = p.vertices();
                let first = points[0].dot(line);
                let (lower, upper) = points[1..].iter().fold((first, first), |(lo, hi), v| {
                    let value = v.dot(line);
                    (lo.min(value), hi.max(value))
                });
                (upper - lower) + 2.0 * p.radius
            }
            Self::Segment(s) | Self::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                (s.point2.dot(line) - s.point1.dot(line)).abs()
            }
        }
    }
}
