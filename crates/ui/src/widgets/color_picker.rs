//! Color picker widget supporting RGBA, HSVA, and hexadecimal conversions.

use korppi_ecs::Component;

/// Color picker widget component with RGBA, HSV, and palette preset storage.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorPicker {
    /// Normalized RGBA color values $[0.0, 1.0]$.
    color: [f32; 4],
    /// Saved swatch palette presets.
    presets: Vec<[f32; 4]>,
}

impl Default for ColorPicker {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            presets: Vec::new(),
        }
    }
}

impl Component for ColorPicker {}

impl ColorPicker {
    /// Creates a new `ColorPicker` initialized to white.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current RGBA color.
    #[inline]
    pub fn color(&self) -> [f32; 4] {
        self.color
    }

    /// Sets the RGBA color directly.
    pub fn set_rgba(&mut self, rgba: [f32; 4]) {
        self.color = [
            rgba[0].clamp(0.0, 1.0),
            rgba[1].clamp(0.0, 1.0),
            rgba[2].clamp(0.0, 1.0),
            rgba[3].clamp(0.0, 1.0),
        ];
    }

    /// Converts current color to HSVA `(hue in [0, 360], sat in [0, 1], val in [0, 1], alpha in [0, 1])`.
    pub fn to_hsva(&self) -> (f32, f32, f32, f32) {
        let (r, g, b, a) = (self.color[0], self.color[1], self.color[2], self.color[3]);
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let v = max;
        let s = if max > 1e-6 { delta / max } else { 0.0 };

        let mut h = if delta.abs() < 1e-6 {
            0.0
        } else if (max - r).abs() < 1e-6 {
            60.0 * (((g - b) / delta) % 6.0)
        } else if (max - g).abs() < 1e-6 {
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            60.0 * (((r - g) / delta) + 4.0)
        };

        if h < 0.0 {
            h += 360.0;
        }

        (h, s, v, a)
    }

    /// Sets the color from HSVA `(h in [0, 360], s in [0, 1], v in [0, 1], a in [0, 1])`.
    pub fn set_hsva(&mut self, h: f32, s: f32, v: f32, a: f32) {
        let h_norm = (h % 360.0 + 360.0) % 360.0;
        let s_clamped = s.clamp(0.0, 1.0);
        let v_clamped = v.clamp(0.0, 1.0);
        let a_clamped = a.clamp(0.0, 1.0);

        let c = v_clamped * s_clamped;
        let h_prime = h_norm / 60.0;
        let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
        let m = v_clamped - c;

        let (r_prime, g_prime, b_prime) = match h_prime as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };

        self.color = [
            (r_prime + m).clamp(0.0, 1.0),
            (g_prime + m).clamp(0.0, 1.0),
            (b_prime + m).clamp(0.0, 1.0),
            a_clamped,
        ];
    }

    /// Returns hexadecimal representation (e.g. `"#FFFFFFFF"`).
    pub fn to_hex_string(&self) -> String {
        let r = (self.color[0] * 255.0).round() as u8;
        let g = (self.color[1] * 255.0).round() as u8;
        let b = (self.color[2] * 255.0).round() as u8;
        let a = (self.color[3] * 255.0).round() as u8;
        format!("#{:02X}{:02X}{:02X}{:02X}", r, g, b, a)
    }

    /// Parses and sets color from a hex string (e.g. `"#RRGGBB"` or `"#RRGGBBAA"`).
    pub fn set_from_hex(&mut self, hex: &str) -> bool {
        let clean = hex.trim().trim_start_matches('#');
        if clean.len() == 6 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&clean[0..2], 16),
                u8::from_str_radix(&clean[2..4], 16),
                u8::from_str_radix(&clean[4..6], 16),
            ) {
                self.color = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0];
                return true;
            }
        } else if clean.len() == 8 {
            if let (Ok(r), Ok(g), Ok(b), Ok(a)) = (
                u8::from_str_radix(&clean[0..2], 16),
                u8::from_str_radix(&clean[2..4], 16),
                u8::from_str_radix(&clean[4..6], 16),
                u8::from_str_radix(&clean[6..8], 16),
            ) {
                self.color = [
                    r as f32 / 255.0,
                    g as f32 / 255.0,
                    b as f32 / 255.0,
                    a as f32 / 255.0,
                ];
                return true;
            }
        }
        false
    }

    /// Returns list of saved palette presets.
    #[inline]
    pub fn presets(&self) -> &[[f32; 4]] {
        &self.presets
    }

    /// Appends a new color to the saved presets palette.
    pub fn add_preset(&mut self, rgba: [f32; 4]) {
        self.presets.push(rgba);
    }
}
