//! Tonemapping operators, exposure control, and color grading (`RND-15`, Probe `G13`).

use korppi_math::Vec3;

/// Available HDR-to-LDR tonemapping operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TonemapOperator {
    /// Extended Reinhard luminance-preserving compression.
    Reinhard,
    /// Academy Color Encoding System (ACES) fitted filmic S-curve.
    #[default]
    Aces,
    /// AgX modern wide-gamut minimum-hue-skew tone mapping.
    AgX,
    /// John Hable's Uncharted 2 filmic curve.
    Filmic,
}

/// Configuration settings for the post-process tonemapping stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TonemapConfig {
    /// Active tonemapping curve operator.
    pub operator: TonemapOperator,
    /// Linear exposure multiplier ($\text{intensity} \times \text{exposure}$).
    pub exposure: f32,
    /// Gamma correction exponent (typically 2.2).
    pub gamma: f32,
    /// Reinhard white point luminance limit.
    pub white_point: f32,
}

impl Default for TonemapConfig {
    fn default() -> Self {
        Self {
            operator: TonemapOperator::Aces,
            exposure: 1.0,
            gamma: 2.2,
            white_point: 10.0,
        }
    }
}

/// Multiplies HDR linear color by an exposure multiplier.
#[inline]
pub fn apply_exposure(color: Vec3, exposure: f32) -> Vec3 {
    color * exposure.max(0.0)
}

/// Extended luminance Reinhard tone reproduction operator.
///
/// Compresses scene luminance according to:
/// $$L_{out} = \frac{L (1 + L / L_{white}^2)}{1 + L}$$
pub fn reinhard_luminance(color: Vec3, white_point: f32) -> Vec3 {
    let c = color.max(Vec3::ZERO);
    let l = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    if l <= 1e-6 {
        return Vec3::ZERO;
    }

    let w2 = (white_point * white_point).max(1e-4);
    let l_out = (l * (1.0 + l / w2)) / (1.0 + l);
    (c * (l_out / l)).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Stephen Hill & Krzysztof Narkowicz fitted ACES filmic curve.
///
/// Maps unbounded linear HDR radiance to $[0.0, 1.0]$ with an S-shaped contrast curve.
pub fn aces_fitted(color: Vec3) -> Vec3 {
    let v = color.max(Vec3::ZERO);
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    let numerator = v * (a * v + Vec3::splat(b));
    let denominator = v * (c * v + Vec3::splat(d)) + Vec3::splat(e);

    (numerator / denominator).clamp(Vec3::ZERO, Vec3::ONE)
}

/// John Hable (Uncharted 2) filmic tone reproduction curve.
pub fn filmic_hable(color: Vec3) -> Vec3 {
    #[inline]
    fn hable_part(x: Vec3) -> Vec3 {
        let a = 0.15;
        let b = 0.50;
        let c = 0.10;
        let d = 0.20;
        let e = 0.02;
        let f = 0.30;
        ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - Vec3::splat(e / f)
    }

    let v = color.max(Vec3::ZERO) * 2.0;
    let curr = hable_part(v);
    let white_scale = 1.0 / hable_part(Vec3::splat(11.2)).x;
    (curr * white_scale).clamp(Vec3::ZERO, Vec3::ONE)
}

/// AgX tone mapping reproduction algorithm minimizing hue shifts and gamut clipping (`RND-15`, Probe `G13`).
///
/// Uses Troy Sobotka's log2 dynamic range $[-12.47393, +4.026069]$ EV and 6th-order polynomial
/// (including the $-0.00232$ constant term). Note that the AgX polynomial already outputs
/// display-referred values.
pub fn agx_tonemap(color: Vec3) -> Vec3 {
    let v = color.max(Vec3::ZERO);

    // 1. AgX Inset Matrix (Rec.709 linear to AgX inset primaries)
    let r = 0.842_479_06 * v.x + 0.078_433_6 * v.y + 0.079_223_75 * v.z;
    let g = 0.042_328_24 * v.x + 0.878_468_63 * v.y + 0.079_166_13 * v.z;
    let b = 0.042_375_65 * v.x + 0.078_433_6 * v.y + 0.879_143 * v.z;
    let agx_in = Vec3::new(r, g, b).max(Vec3::splat(1e-10));

    // 2. Log2 encoding: Troy Sobotka's AgX dynamic range [-12.47393, +4.026069] EV
    let min_ev = -12.473_93f32;
    let max_ev = 4.026_069f32;
    let log_x = ((agx_in.x.log2() - min_ev) / (max_ev - min_ev)).clamp(0.0, 1.0);
    let log_y = ((agx_in.y.log2() - min_ev) / (max_ev - min_ev)).clamp(0.0, 1.0);
    let log_z = ((agx_in.z.log2() - min_ev) / (max_ev - min_ev)).clamp(0.0, 1.0);

    // 3. AgX default contrast polynomial: 6th-order approximation with constant term
    let apply_contrast = |x: f32| -> f32 {
        let x2 = x * x;
        let x4 = x2 * x2;
        (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
            - 0.00232)
            .clamp(0.0, 1.0)
    };

    let cx = apply_contrast(log_x);
    let cy = apply_contrast(log_y);
    let cz = apply_contrast(log_z);

    // 4. AgX Outset Matrix back to Rec.709 display primaries
    let out_r = 1.196_879 * cx - 0.098_020_88 * cy - 0.099_029_74 * cz;
    let out_g = -0.052_896_85 * cx + 1.151_903_1 * cy - 0.098_961_18 * cz;
    let out_b = -0.052_971_64 * cx - 0.098_043_45 * cy + 1.151_073_7 * cz;

    Vec3::new(out_r, out_g, out_b).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Evaluates the complete tonemapping pipeline (exposure -> operator -> display encoding).
///
/// Note: [`TonemapOperator::AgX`] already outputs display-encoded sRGB values (`RND-15`),
/// whereas [`TonemapOperator::Reinhard`], [`TonemapOperator::Aces`], and [`TonemapOperator::Filmic`]
/// output linear LDR values that are then gamma-encoded by `1.0 / config.gamma`.
pub fn evaluate_tonemapping(color: Vec3, config: &TonemapConfig) -> Vec3 {
    let exposed = apply_exposure(color, config.exposure);
    match config.operator {
        TonemapOperator::AgX => agx_tonemap(exposed),
        operator => {
            let mapped = match operator {
                TonemapOperator::Reinhard => reinhard_luminance(exposed, config.white_point),
                TonemapOperator::Aces => aces_fitted(exposed),
                TonemapOperator::Filmic => filmic_hable(exposed),
                TonemapOperator::AgX => unreachable!(),
            };
            let inv_gamma = 1.0 / config.gamma.max(0.001);
            Vec3::new(
                mapped.x.powf(inv_gamma),
                mapped.y.powf(inv_gamma),
                mapped.z.powf(inv_gamma),
            )
            .clamp(Vec3::ZERO, Vec3::ONE)
        }
    }
}
