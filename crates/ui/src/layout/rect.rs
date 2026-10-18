//! 2D bounding rectangle geometry for UI layout and hit testing.

use korppi_math::Vec2;

/// Axis-aligned 2D rectangle in UI coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect2d {
    /// Horizontal origin coordinate.
    pub x: f32,
    /// Vertical origin coordinate.
    pub y: f32,
    /// Width dimension.
    pub width: f32,
    /// Height dimension.
    pub height: f32,
}

impl Rect2d {
    /// Creates a new rectangle with the given origin and dimensions.
    #[inline]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Creates a rectangle bounding the minimum and maximum points.
    #[inline]
    pub fn from_min_max(min: Vec2, max: Vec2) -> Self {
        Self {
            x: min.x,
            y: min.y,
            width: (max.x - min.x).max(0.0),
            height: (max.y - min.y).max(0.0),
        }
    }

    /// Returns the top-left minimum point.
    #[inline]
    pub fn min(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Returns the bottom-right maximum point.
    #[inline]
    pub fn max(&self) -> Vec2 {
        Vec2::new(self.x + self.width, self.y + self.height)
    }

    /// Returns the size vector `(width, height)`.
    #[inline]
    pub fn size(&self) -> Vec2 {
        Vec2::new(self.width, self.height)
    }

    /// Returns the center point of the rectangle.
    #[inline]
    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    /// Tests if a 2D point lies inside the rectangle (inclusive of boundaries).
    #[inline]
    pub fn contains_point(&self, p: Vec2) -> bool {
        p.x >= self.x && p.x <= self.x + self.width && p.y >= self.y && p.y <= self.y + self.height
    }

    /// Tests if this rectangle intersects another rectangle.
    #[inline]
    pub fn intersects(&self, other: &Rect2d) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Computes the intersection rectangle between two rectangles, if overlapping.
    pub fn intersection(&self, other: &Rect2d) -> Option<Rect2d> {
        let min_x = self.x.max(other.x);
        let min_y = self.y.max(other.y);
        let max_x = (self.x + self.width).min(other.x + other.width);
        let max_y = (self.y + self.height).min(other.y + other.height);

        if min_x < max_x && min_y < max_y {
            Some(Rect2d {
                x: min_x,
                y: min_y,
                width: max_x - min_x,
                height: max_y - min_y,
            })
        } else {
            None
        }
    }
}
