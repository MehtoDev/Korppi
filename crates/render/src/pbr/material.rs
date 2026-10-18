//! Surface and material parameters for physically based rendering.

use korppi_math::Vec3;

/// Clear Coat specular resin layer parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearCoat {
    /// Perceptual roughness of the clear coat layer ($\in [0.0, 1.0]$).
    pub roughness: f32,
    /// Layer blending weight ($\in [0.0, 1.0]$).
    pub weight: f32,
}

/// Charlie Sheen grazing lobe parameters for fabric, cloth, and velvet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharlieSheen {
    /// Sheen tint color.
    pub color: Vec3,
    /// Sheen roughness ($\in [0.0, 1.0]$).
    pub roughness: f32,
}

/// Metallic-roughness PBR material parameters with multi-layer extensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbrMaterial {
    /// Perceptual surface roughness ($\in [0.0, 1.0]$).
    pub roughness: f32,
    /// Metallic factor ($0.0 = \text{dielectric}, 1.0 = \text{conductor}$).
    pub metallic: f32,
    /// Specular reflectance for dielectrics at normal incidence (default $0.5 \implies F_0 = 0.04$).
    pub reflectance: f32,
    /// Optional clear coat layer.
    pub clear_coat: Option<ClearCoat>,
    /// Optional Charlie sheen fabric layer.
    pub sheen: Option<CharlieSheen>,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            roughness: 0.5,
            metallic: 0.0,
            reflectance: 0.5,
            clear_coat: None,
            sheen: None,
        }
    }
}

/// Geometric surface state evaluated at a single shading fragment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbrSurface {
    /// World-space position.
    pub position: Vec3,
    /// Unit surface normal.
    pub normal: Vec3,
    /// Unit view vector from surface towards the eye.
    pub view: Vec3,
    /// Base color / albedo.
    pub albedo: Vec3,
    /// Material properties.
    pub material: PbrMaterial,
}
