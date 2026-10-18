#version 450
// Clustered Forward PBR fragment shader (`RND-01`, `RND-06`, `RND-15`).
//
// Evaluates:
// - Multi-scattering GGX microfacet specular with exact 21x21 DFG LUT bilinear interpolation
// - Energy-conserving Lambertian diffuse
// - Clear Coat second specular layer with substrate attenuation
// - Charlie Sheen cloth layer with Neubelt visibility and sheen directional albedo attenuation
// - Clustered omnidirectional point lights and cone spotlights from GPU compute culling SSBOs
// - Unified shadow atlas: 4-cascade stabilized CSM, spotlight perspective shadows, and octahedral
//   omnidirectional point-light shadows with 9-tap PCF filtering
// - Screen-Space Ambient Occlusion (SSAO) modulating ambient irradiance
//
// Compiled with:
//   glslc -O forward_pbr.frag -o forward_pbr.frag.spv

const float PI = 3.14159265358979323846;
const float MIN_PERCEPTUAL_ROUGHNESS = 0.045;
const uint DFG_LUT_DIM = 21u;

struct PointLightGpu {
    vec4 position_radius;
    vec4 color_intensity;
};

struct SpotLightGpu {
    vec4 position_range;
    vec4 direction_half_angle;
    vec4 color_intensity;
};

struct ClusterRecord {
    uint point_offset;
    uint point_count;
    uint spot_offset;
    uint spot_count;
    uint decal_offset;
    uint decal_count;
    uint probe_offset;
    uint probe_count;
};

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 ambient_light;
    vec4 dir_light_dir;
    vec4 dir_light_radiance;
    uvec4 cluster_dims;
    vec4 cluster_depth_params;
    vec4 viewport_and_shadow_counts;
    mat4 csm_view_proj[4];
    vec4 csm_atlas_uv[4];
    vec4 csm_split_depths;
    mat4 spot_shadow_view_proj[4];
    vec4 spot_shadow_atlas_uv[4];
    uvec4 spot_shadow_light_indices;
    vec4 point_shadow_pos_radius[4];
    vec4 point_shadow_atlas_uv[4];
    uvec4 point_shadow_light_indices;
} scene;

layout(set = 0, binding = 1) uniform sampler2D u_shadow_atlas;
layout(set = 0, binding = 2) uniform sampler2D u_ssao;

layout(std430, set = 0, binding = 3) readonly buffer DfgLutBuffer {
    // Each entry: x = dfg_scale, y = dfg_bias, z = sheen_directional_albedo, w = 0
    vec4 lut[];
} dfg_buf;

layout(std430, set = 1, binding = 0) readonly buffer PointLightsBuffer {
    PointLightGpu point_lights[];
};

layout(std430, set = 1, binding = 1) readonly buffer SpotLightsBuffer {
    SpotLightGpu spot_lights[];
};

layout(std430, set = 1, binding = 2) readonly buffer ClusterRecordsBuffer {
    ClusterRecord records[];
};

layout(std430, set = 1, binding = 3) readonly buffer LightIndicesBuffer {
    uint light_indices[];
};

layout(push_constant) uniform DrawPush {
    mat4 model;
    vec4 albedo_roughness;
    vec4 emissive_metallic;
    vec4 reflectance_clearcoat;
    vec4 sheen;
} pc;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_world_normal;
layout(location = 2) in vec3 v_view_pos;
layout(location = 3) in vec3 v_view_normal;
layout(location = 4) in vec2 v_uv;

layout(location = 0) out vec4 o_hdr_color;

vec3 sample_dfg_lut(float n_dot_v, float perceptual_roughness) {
    float nv = clamp(n_dot_v, 0.0, 1.0);
    float r = clamp(perceptual_roughness, 0.0, 1.0);
    float max_idx = float(DFG_LUT_DIM - 1u);

    float fx = nv * max_idx;
    float fy = r * max_idx;
    uint x0 = min(uint(fx), DFG_LUT_DIM - 1u);
    uint y0 = min(uint(fy), DFG_LUT_DIM - 1u);
    uint x1 = min(x0 + 1u, DFG_LUT_DIM - 1u);
    uint y1 = min(y0 + 1u, DFG_LUT_DIM - 1u);
    float tx = fx - float(x0);
    float ty = fy - float(y0);

    vec3 c00 = dfg_buf.lut[y0 * DFG_LUT_DIM + x0].xyz;
    vec3 c10 = dfg_buf.lut[y0 * DFG_LUT_DIM + x1].xyz;
    vec3 c01 = dfg_buf.lut[y1 * DFG_LUT_DIM + x0].xyz;
    vec3 c11 = dfg_buf.lut[y1 * DFG_LUT_DIM + x1].xyz;

    vec3 c0 = mix(c00, c10, tx);
    vec3 c1 = mix(c01, c11, tx);
    return mix(c0, c1, ty);
}

float d_ggx(float n_dot_h, float roughness) {
    float r = clamp(roughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    float alpha = r * r;
    float a2 = alpha * alpha;
    float nh = clamp(n_dot_h, 0.0, 1.0);
    float denom_part = nh * nh * (a2 - 1.0) + 1.0;
    float denom = PI * denom_part * denom_part;
    return a2 / max(denom, 1e-7);
}

float v_smith_heitz(float n_dot_l, float n_dot_v, float roughness) {
    float r = clamp(roughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    float alpha = r * r;
    float a2 = alpha * alpha;
    float nl = clamp(n_dot_l, 0.0, 1.0);
    float nv = clamp(n_dot_v, 0.0, 1.0);

    float lambda_v = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    float lambda_l = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1e-7);
}

float v_neubelt(float n_dot_l, float n_dot_v) {
    float nl = clamp(n_dot_l, 0.0, 1.0);
    float nv = clamp(n_dot_v, 0.0, 1.0);
    return 1.0 / max(4.0 * (nl + nv - nl * nv), 1e-5);
}

vec3 fresnel_schlick(float v_dot_h, vec3 f0) {
    float vh = clamp(v_dot_h, 0.0, 1.0);
    float factor = pow(1.0 - vh, 5.0);
    return f0 + (vec3(1.0) - f0) * factor;
}

float d_charlie(float n_dot_h, float roughness) {
    float alpha = clamp(roughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    float nh = clamp(n_dot_h, 0.0, 1.0);
    float sin2 = max(1.0 - nh * nh, 0.0);
    float cos2 = nh * nh;
    float denom = sin2 + alpha * alpha * cos2;
    if (denom <= 0.0) {
        return 0.0;
    }
    float inv_alpha = 1.0 / alpha;
    return (2.0 + inv_alpha) * pow(max(sin2, 1e-8), inv_alpha * 0.5) / (2.0 * PI);
}

vec3 evaluate_pbr_lobe(vec3 n, vec3 v, vec3 l, vec3 radiance, vec2 dfg) {
    float n_dot_l = max(dot(n, l), 0.0);
    float n_dot_v = max(dot(n, v), 1e-4);
    if (n_dot_l <= 0.0) {
        return vec3(0.0);
    }

    vec3 h = normalize(v + l);
    float n_dot_h = max(dot(n, h), 0.0);
    float v_dot_h = max(dot(v, h), 0.0);

    vec3 albedo = pc.albedo_roughness.rgb;
    float roughness = pc.albedo_roughness.w;
    float metallic = pc.emissive_metallic.w;
    float reflectance = pc.reflectance_clearcoat.x;

    vec3 dielectric_f0 = vec3(0.08 * reflectance);
    vec3 f0 = mix(dielectric_f0, albedo, metallic);

    float d = d_ggx(n_dot_h, roughness);
    float vis = v_smith_heitz(n_dot_l, n_dot_v, roughness);
    vec3 f = fresnel_schlick(v_dot_h, f0);

    float dir_albedo = max(dfg.x + dfg.y, 1e-4);
    vec3 energy_comp = vec3(1.0) + f0 * (1.0 / dir_albedo - 1.0);
    vec3 specular = f * (d * vis) * energy_comp;

    vec3 spec_dir_albedo = clamp((f0 * dfg.x + vec3(dfg.y)) * energy_comp, vec3(0.0), vec3(1.0));
    vec3 kd = min(vec3(1.0) - f, vec3(1.0) - spec_dir_albedo) * (1.0 - metallic);
    vec3 diffuse = (albedo / PI) * kd;

    vec3 composite = diffuse + specular;

    if (pc.reflectance_clearcoat.w > 0.5) {
        float cc_roughness = pc.reflectance_clearcoat.y;
        float cc_weight = pc.reflectance_clearcoat.z;
        vec3 cc_f = fresnel_schlick(v_dot_h, vec3(0.04));
        float cc_d = d_ggx(n_dot_h, cc_roughness);
        float cc_vis = v_smith_heitz(n_dot_l, n_dot_v, cc_roughness);
        vec3 cc_spec = cc_f * (cc_d * cc_vis * cc_weight);
        composite = composite * (vec3(1.0) - cc_f * cc_weight) + cc_spec;
    }

    if (pc.sheen.w > 0.0) {
        vec3 sheen_color = pc.sheen.rgb;
        float sheen_roughness = pc.sheen.w;
        float sh_d = d_charlie(n_dot_h, sheen_roughness);
        float sh_vis = v_neubelt(n_dot_l, n_dot_v);
        vec3 sh_lobe = sheen_color * (sh_d * sh_vis);
        float max_sheen = clamp(max(sheen_color.r, max(sheen_color.g, sheen_color.b)), 0.0, 1.0);
        float sheen_albedo = sample_dfg_lut(n_dot_v, sheen_roughness).z;
        float sheen_scaling = clamp(1.0 - max_sheen * sheen_albedo, 0.0, 1.0);
        composite = composite * sheen_scaling + sh_lobe;
    }

    return composite * radiance * n_dot_l;
}

vec2 encode_octahedral(vec3 dir) {
    vec3 d = normalize(dir);
    float l1 = abs(d.x) + abs(d.y) + abs(d.z);
    vec2 res = d.xy / max(l1, 1e-6);
    if (d.z < 0.0) {
        float rx = res.x;
        float ry = res.y;
        res.x = (1.0 - abs(ry)) * (step(0.0, rx) * 2.0 - 1.0);
        res.y = (1.0 - abs(rx)) * (step(0.0, ry) * 2.0 - 1.0);
    }
    return res * 0.5 + 0.5;
}

float pcf_sample_tile(vec2 local_uv, vec4 tile_uv_rect, float receiver_depth, float bias) {
    float texel = scene.cluster_depth_params.w;
    vec2 tile_min = tile_uv_rect.xy + vec2(texel * 1.5);
    vec2 tile_max = tile_uv_rect.xy + tile_uv_rect.zw - vec2(texel * 1.5);
    vec2 base_uv = tile_uv_rect.xy + local_uv * tile_uv_rect.zw;

    float vis = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec2 sample_uv = clamp(base_uv + vec2(float(dx), float(dy)) * texel, tile_min, tile_max);
            float blocker_depth = texture(u_shadow_atlas, sample_uv).r;
            if (receiver_depth - bias <= blocker_depth) {
                vis += 1.0;
            }
        }
    }
    return vis / 9.0;
}

float sample_csm_shadow(vec3 world_pos, vec3 world_normal, float linear_z) {
    int cascade_idx = 3;
    for (int i = 0; i < 4; ++i) {
        if (linear_z <= scene.csm_split_depths[i]) {
            cascade_idx = i;
            break;
        }
    }

    vec3 biased_pos = world_pos + world_normal * 0.015;
    vec4 clip = scene.csm_view_proj[cascade_idx] * vec4(biased_pos, 1.0);
    if (abs(clip.w) <= 1e-6) {
        return 1.0;
    }
    vec3 ndc = clip.xyz / clip.w;
    vec2 local_uv = vec2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y));
    if (any(lessThan(vec3(local_uv, ndc.z), vec3(0.01, 0.01, 0.0)))) {
        return 1.0;
    }
    if (any(greaterThan(vec3(local_uv, ndc.z), vec3(0.99, 0.99, 1.0)))) {
        return 1.0;
    }
    return pcf_sample_tile(local_uv, scene.csm_atlas_uv[cascade_idx], ndc.z, 0.0025);
}

float sample_spot_shadow(uint light_idx, vec3 world_pos, vec3 world_normal) {
    uint count = uint(scene.viewport_and_shadow_counts.z + 0.5);
    for (uint s = 0u; s < count; ++s) {
        if (scene.spot_shadow_light_indices[s] == light_idx) {
            vec3 biased_pos = world_pos + world_normal * 0.015;
            vec4 clip = scene.spot_shadow_view_proj[s] * vec4(biased_pos, 1.0);
            if (clip.w <= 1e-5) {
                return 1.0;
            }
            vec3 ndc = clip.xyz / clip.w;
            vec2 local_uv = vec2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y));
            if (any(lessThan(vec3(local_uv, ndc.z), vec3(0.01, 0.01, 0.0)))) {
                return 1.0;
            }
            if (any(greaterThan(vec3(local_uv, ndc.z), vec3(0.99, 0.99, 1.0)))) {
                return 1.0;
            }
            return pcf_sample_tile(local_uv, scene.spot_shadow_atlas_uv[s], ndc.z, 0.003);
        }
    }
    return 1.0;
}

float sample_point_shadow(uint light_idx, vec3 world_pos, vec3 world_normal) {
    uint count = uint(scene.viewport_and_shadow_counts.w + 0.5);
    for (uint p = 0u; p < count; ++p) {
        if (scene.point_shadow_light_indices[p] == light_idx) {
            vec3 light_pos = scene.point_shadow_pos_radius[p].xyz;
            float radius = max(scene.point_shadow_pos_radius[p].w, 1e-4);
            vec3 biased_pos = world_pos + world_normal * 0.02;
            vec3 to_frag = biased_pos - light_pos;
            float dist = length(to_frag);
            if (dist >= radius) {
                return 1.0;
            }
            if (dist <= 1e-5) {
                return 1.0;
            }
            vec2 local_uv = encode_octahedral(to_frag / dist);
            float receiver_depth = dist / radius;
            return pcf_sample_tile(local_uv, scene.point_shadow_atlas_uv[p], receiver_depth, 0.008);
        }
    }
    return 1.0;
}

void main() {
    vec3 n_ws = normalize(v_world_normal);
    vec3 v_ws = normalize(scene.camera_pos.xyz - v_world_pos);
    vec3 n_vs = normalize(v_view_normal);
    vec3 v_vs = normalize(-v_view_pos);

    float n_dot_v = max(dot(n_ws, v_ws), 1e-4);
    vec2 dfg = sample_dfg_lut(n_dot_v, pc.albedo_roughness.w).xy;

    vec2 screen_uv = gl_FragCoord.xy * scene.viewport_and_shadow_counts.xy;
    float ao = texture(u_ssao, screen_uv).r;

    vec3 total_hdr = pc.albedo_roughness.rgb * scene.ambient_light.rgb * ao + pc.emissive_metallic.rgb;

    float linear_z = max(-v_view_pos.z, scene.cluster_depth_params.x);

    // 1. Directional light + CSM shadows
    if (scene.dir_light_dir.w > 0.5) {
        vec3 l_ws = normalize(-scene.dir_light_dir.xyz);
        float shadow_vis = 1.0;
        if (scene.dir_light_radiance.w > 0.5) {
            shadow_vis = sample_csm_shadow(v_world_pos, n_ws, linear_z);
        }
        if (shadow_vis > 0.0) {
            total_hdr += evaluate_pbr_lobe(n_ws, v_ws, l_ws, scene.dir_light_radiance.rgb * shadow_vis, dfg);
        }
    }

    // 2. Clustered point and spot lights
    uint grid_x = max(scene.cluster_dims.x, 1u);
    uint grid_y = max(scene.cluster_dims.y, 1u);
    uint grid_z = max(scene.cluster_dims.z, 1u);
    uint tile_px = max(scene.cluster_dims.w, 1u);

    uint tile_x = min(uint(max(gl_FragCoord.x, 0.0)) / tile_px, grid_x - 1u);
    uint tile_y = min(uint(max(gl_FragCoord.y, 0.0)) / tile_px, grid_y - 1u);
    float log_ratio = log(max(linear_z / scene.cluster_depth_params.x, 1.0)) * scene.cluster_depth_params.z;
    uint slice_z = uint(clamp(floor(log_ratio * float(grid_z)), 0.0, float(grid_z - 1u)));
    uint cluster_idx = slice_z * (grid_x * grid_y) + tile_y * grid_x + tile_x;

    ClusterRecord rec = records[cluster_idx];

    for (uint i = 0u; i < rec.point_count; ++i) {
        uint pt_idx = light_indices[rec.point_offset + i];
        PointLightGpu pt = point_lights[pt_idx];
        vec3 to_light = pt.position_radius.xyz - v_view_pos;
        float dist = length(to_light);
        float radius = pt.position_radius.w;
        if (dist < radius) {
            if (dist > 1e-5) {
                float shadow_vis = sample_point_shadow(pt_idx, v_world_pos, n_ws);
                if (shadow_vis > 0.0) {
                    float u = clamp(dist / radius, 0.0, 1.0);
                    float win = 1.0 - u * u * u * u;
                    float atten = (win * win) / (dist * dist + 1.0);
                    vec3 l_vs = to_light / dist;
                    vec3 radiance = pt.color_intensity.rgb * (pt.color_intensity.a * atten * shadow_vis);
                    total_hdr += evaluate_pbr_lobe(n_vs, v_vs, l_vs, radiance, dfg);
                }
            }
        }
    }

    for (uint j = 0u; j < rec.spot_count; ++j) {
        uint sp_idx = light_indices[rec.spot_offset + j];
        SpotLightGpu sp = spot_lights[sp_idx];
        vec3 to_light = sp.position_range.xyz - v_view_pos;
        float dist = length(to_light);
        float range = sp.position_range.w;
        if (dist < range) {
            if (dist > 1e-5) {
                vec3 l_vs = to_light / dist;
                float cos_theta = dot(-l_vs, normalize(sp.direction_half_angle.xyz));
                float cos_outer = cos(sp.direction_half_angle.w);
                float cos_inner = cos(sp.direction_half_angle.w * 0.75);
                if (cos_theta > cos_outer) {
                    float shadow_vis = sample_spot_shadow(sp_idx, v_world_pos, n_ws);
                    if (shadow_vis > 0.0) {
                        float cone_atten = clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
                        cone_atten = cone_atten * cone_atten;
                        float u = clamp(dist / range, 0.0, 1.0);
                        float win = 1.0 - u * u * u * u;
                        float dist_atten = (win * win) / (dist * dist + 1.0);
                        vec3 radiance = sp.color_intensity.rgb * (sp.color_intensity.a * dist_atten * cone_atten * shadow_vis);
                        total_hdr += evaluate_pbr_lobe(n_vs, v_vs, l_vs, radiance, dfg);
                    }
                }
            }
        }
    }

    o_hdr_color = vec4(max(total_hdr, vec3(0.0)), 1.0);
}
