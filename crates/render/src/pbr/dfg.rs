//! Pre-integrated Directional Filtered Geometry (DFG) split-sum and Charlie sheen LUT (`RND-06`, `RND-15`).

use std::f32::consts::PI;
use std::sync::OnceLock;

use crate::pbr::brdf::{d_charlie, v_neubelt, v_smith_heitz, MIN_PERCEPTUAL_ROUGHNESS};
use korppi_math::{Vec2, Vec3};

/// Dimension of the 2D DFG lookup table along each axis (`N dot V` and perceptual `roughness`).
pub const DFG_LUT_DIM: usize = 21;

/// Radical inverse base-2 (Van der Corput sequence) for low-discrepancy Hammersley sampling.
#[inline]
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.swap_bytes();
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    (bits as f32) * 2.328_306_4e-10
}

/// Importance-samples the GGX NDF around surface normal `N = (0, 1, 0)` for perceptual `roughness`.
#[inline]
fn importance_sample_ggx(xi_x: f32, xi_y: f32, roughness: f32) -> Vec3 {
    let r = roughness.clamp(MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    let alpha = r * r;
    let a2 = alpha * alpha;

    let phi = 2.0 * PI * xi_x;
    let cos_theta = ((1.0 - xi_y) / (1.0 + (a2 - 1.0) * xi_y))
        .clamp(0.0, 1.0)
        .sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin())
}

/// Numerically integrates the split-sum BRDF directional albedo terms `(scale, bias)` = `(A, B)`
/// for GGX + Heitz height-correlated Smith visibility using `sample_count` Hammersley samples (`RND-06`).
///
/// The single-scattering directional albedo for normal-incidence reflectance $F_0$ is:
/// $$E_{\text{ss}}(\mu, r) = F_0 \cdot A + B$$
pub fn integrate_dfg(n_dot_v: f32, roughness: f32, sample_count: u32) -> Vec2 {
    let nv = n_dot_v.clamp(1e-4, 1.0);
    let r = roughness.clamp(MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    let v = Vec3::new((1.0 - nv * nv).max(0.0).sqrt(), nv, 0.0);

    let mut a = 0.0f64;
    let mut b = 0.0f64;
    let count = sample_count.max(1);

    for i in 0..count {
        let xi_x = (i as f32 + 0.5) / (count as f32);
        let xi_y = radical_inverse_vdc(i);
        let h = importance_sample_ggx(xi_x, xi_y, r);
        let v_dot_h = v.dot(h).max(0.0);
        let l = (h * (2.0 * v_dot_h) - v).normalize();

        let n_dot_l = l.y.max(0.0);
        let n_dot_h = h.y.max(1e-5);

        if n_dot_l > 0.0 {
            let vis = v_smith_heitz(n_dot_l, nv, r);
            let g_vis = (4.0 * vis * n_dot_l * v_dot_h / n_dot_h).max(0.0);
            let fc = (1.0 - v_dot_h).powi(5);

            a += f64::from(g_vis * (1.0 - fc));
            b += f64::from(g_vis * fc);
        }
    }

    let inv_n = 1.0 / f64::from(count);
    Vec2::new(
        (a * inv_n).clamp(0.0, 1.0) as f32,
        (b * inv_n).clamp(0.0, 1.0) as f32,
    )
}

/// Integrates `(scale, bias, charlie_sheen)` over the upper hemisphere.
fn integrate_lut_entry(n_dot_v: f32, roughness: f32) -> Vec3 {
    let nv = n_dot_v.clamp(1e-4, 1.0);
    let r = roughness.clamp(MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    let v = Vec3::new((1.0 - nv * nv).max(0.0).sqrt(), nv, 0.0);

    // Integrate Charlie sheen directional albedo by midpoint hemisphere quadrature
    let (nu, nphi) = (64, 32);
    let mut sheen = 0.0f64;
    for i in 0..nu {
        let cos_t = (i as f32 + 0.5) / (nu as f32);
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        for j in 0..nphi {
            let phi = (j as f32 + 0.5) / (nphi as f32) * 2.0 * PI;
            let l = Vec3::new(sin_t * phi.cos(), cos_t, sin_t * phi.sin());
            let h = (v + l).normalize();
            let nh = h.y.max(0.0);
            sheen += f64::from(d_charlie(nh, r) * v_neubelt(cos_t, nv) * cos_t);
        }
    }
    let w = (1.0 / (nu as f64)) * (2.0 * std::f64::consts::PI / (nphi as f64));
    let sheen_albedo = ((sheen * w) as f32 + 2e-3).clamp(0.0, 1.0);

    let imp = integrate_dfg(nv, r, 1024);
    let bias = imp.y.clamp(1e-5, 1.0);
    let scale = (imp.x + 2e-3).clamp(1e-5, (1.0 - bias).max(1e-5));

    Vec3::new(scale, bias, sheen_albedo)
}

fn dfg_lut() -> &'static [[Vec3; DFG_LUT_DIM]; DFG_LUT_DIM] {
    static LUT: OnceLock<[[Vec3; DFG_LUT_DIM]; DFG_LUT_DIM]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [[Vec3::ZERO; DFG_LUT_DIM]; DFG_LUT_DIM];
        let max_idx = (DFG_LUT_DIM - 1) as f32;
        for (y, row) in table.iter_mut().enumerate() {
            let roughness = (y as f32 / max_idx).max(MIN_PERCEPTUAL_ROUGHNESS);
            for (x, cell) in row.iter_mut().enumerate() {
                let n_dot_v = (x as f32 / max_idx).max(1e-3);
                *cell = integrate_lut_entry(n_dot_v, roughness);
            }
        }
        table
    })
}

fn sample_lut(n_dot_v: f32, roughness: f32) -> Vec3 {
    let table = dfg_lut();
    let max_idx = (DFG_LUT_DIM - 1) as f32;

    let nv = n_dot_v.clamp(0.0, 1.0);
    let r = roughness.clamp(0.0, 1.0);

    let u = nv * max_idx;
    let v = r * max_idx;

    let x0 = (u.floor() as usize).min(DFG_LUT_DIM - 1);
    let y0 = (v.floor() as usize).min(DFG_LUT_DIM - 1);
    let x1 = (x0 + 1).min(DFG_LUT_DIM - 1);
    let y1 = (y0 + 1).min(DFG_LUT_DIM - 1);

    let tx = u - (x0 as f32);
    let ty = v - (y0 as f32);

    let c00 = table[y0][x0];
    let c10 = table[y0][x1];
    let c01 = table[y1][x0];
    let c11 = table[y1][x1];

    let top = c00.lerp(c10, tx);
    let bottom = c01.lerp(c11, tx);
    top.lerp(bottom, ty)
}

/// Evaluates the 2D split-sum BRDF integration `(scale, bias)` = `(A, B)` for GGX specular reflectance (`RND-06`).
///
/// Bilinearly interpolates a pre-integrated $21 \times 21$ DFG lookup table (step $0.05$) computed directly from
/// [`d_ggx`](crate::pbr::d_ggx) and [`v_smith_heitz`](crate::pbr::v_smith_heitz):
/// $$\text{DirectionalAlbedo} = F_0 \cdot \text{scale} + \text{bias}$$
pub fn approximate_dfg(n_dot_v: f32, roughness: f32) -> Vec2 {
    let s = sample_lut(n_dot_v, roughness);
    Vec2::new(s.x, s.y)
}

/// Evaluates the pre-integrated directional albedo of the Charlie sheen lobe (`RND-15`).
pub fn approximate_sheen_dfg(n_dot_v: f32, roughness: f32) -> f32 {
    sample_lut(n_dot_v, roughness).z
}

/// Returns the flattened `21 * 21` DFG and Charlie sheen lookup table as `[scale, bias, sheen_albedo, 0.0]`
/// rows for uploading to a GPU buffer or texture (`FIX-6.4`).
pub fn dfg_lut_packed() -> [[f32; 4]; DFG_LUT_DIM * DFG_LUT_DIM] {
    let table = dfg_lut();
    let mut packed = [[0.0f32; 4]; DFG_LUT_DIM * DFG_LUT_DIM];
    for (y, row) in table.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            packed[y * DFG_LUT_DIM + x] = [cell.x, cell.y, cell.z, 0.0];
        }
    }
    packed
}
