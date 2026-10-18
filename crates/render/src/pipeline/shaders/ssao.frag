#version 450
// Screen-Space Ambient Occlusion (SSAO) pass reconstructing view-space position and normal from
// the 1x Reverse-Z depth prepass texture (`RND-01`).
//
// Compiled with:
//   glslc -O ssao.frag -o ssao.frag.spv

layout(set = 0, binding = 0) uniform SsaoParams {
    mat4 proj;
    mat4 inv_proj;
    // x = sample_radius, y = bias, z = intensity, w = power
    vec4 params0;
    // x = inv_width, y = inv_height, z = float(kernel_size), w = 0
    vec4 params1;
    vec4 samples[64];
} ubo;

layout(set = 0, binding = 1) uniform sampler2D u_depth;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_ao;

vec3 reconstruct_view_pos(vec2 uv, float depth) {
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec4 clip = vec4(ndc, depth, 1.0);
    vec4 v = ubo.inv_proj * clip;
    return v.xyz / max(abs(v.w), 1e-6);
}

void main() {
    float center_depth = texture(u_depth, v_uv).r;
    if (center_depth <= 1e-6) {
        o_ao = vec4(1.0);
        return;
    }

    vec2 texel = ubo.params1.xy;
    vec3 p_c = reconstruct_view_pos(v_uv, center_depth);

    float d_r = texture(u_depth, clamp(v_uv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))).r;
    float d_l = texture(u_depth, clamp(v_uv - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))).r;
    float d_u = texture(u_depth, clamp(v_uv - vec2(0.0, texel.y), vec2(0.0), vec2(1.0))).r;
    float d_d = texture(u_depth, clamp(v_uv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0))).r;

    vec3 p_r = reconstruct_view_pos(v_uv + vec2(texel.x, 0.0), d_r);
    vec3 p_l = reconstruct_view_pos(v_uv - vec2(texel.x, 0.0), d_l);
    vec3 p_u = reconstruct_view_pos(v_uv - vec2(0.0, texel.y), d_u);
    vec3 p_d = reconstruct_view_pos(v_uv + vec2(0.0, texel.y), d_d);

    vec3 dx;
    if (abs(p_r.z - p_c.z) < abs(p_c.z - p_l.z)) {
        dx = p_r - p_c;
    } else {
        dx = p_c - p_l;
    }
    vec3 dy;
    if (abs(p_u.z - p_c.z) < abs(p_c.z - p_d.z)) {
        dy = p_u - p_c;
    } else {
        dy = p_c - p_d;
    }

    vec3 n = cross(dx, dy);
    float n_len = length(n);
    if (n_len <= 1e-6) {
        n = vec3(0.0, 0.0, 1.0);
    } else {
        n /= n_len;
        if (n.z < 0.0) {
            n = -n;
        }
    }

    // Deterministic per-pixel rotation angle in screen space (no temporal jitter, Rule 3)
    float angle = fract(52.9829189 * fract(dot(floor(gl_FragCoord.xy), vec2(0.06711056, 0.00583715)))) * 6.2831853;
    vec3 rand_vec = vec3(cos(angle), sin(angle), 0.0);
    vec3 tangent = rand_vec - n * dot(rand_vec, n);
    float t_len = length(tangent);
    if (t_len <= 1e-5) {
        if (abs(n.x) < 0.9) {
            tangent = normalize(cross(vec3(1.0, 0.0, 0.0), n));
        } else {
            tangent = normalize(cross(vec3(0.0, 1.0, 0.0), n));
        }
    } else {
        tangent /= t_len;
    }
    vec3 bitangent = cross(n, tangent);
    mat3 tbn = mat3(tangent, bitangent, n);

    float radius = max(ubo.params0.x, 0.001);
    float bias = ubo.params0.y;
    float intensity = ubo.params0.z;
    float power = max(ubo.params0.w, 0.01);
    int kernel_size = clamp(int(ubo.params1.z + 0.5), 1, 64);

    float occlusion = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        vec3 s_pos = p_c + (tbn * ubo.samples[i].xyz) * radius;
        vec4 offset_clip = ubo.proj * vec4(s_pos, 1.0);
        if (offset_clip.w <= 1e-5) {
            continue;
        }
        vec2 offset_ndc = offset_clip.xy / offset_clip.w;
        vec2 s_uv = vec2(0.5 * (offset_ndc.x + 1.0), 0.5 * (1.0 - offset_ndc.y));
        if (any(lessThan(s_uv, vec2(0.0)))) {
            continue;
        }
        if (any(greaterThan(s_uv, vec2(1.0)))) {
            continue;
        }
        float occ_depth = texture(u_depth, s_uv).r;
        if (occ_depth <= 1e-6) {
            continue;
        }
        vec3 occ_pos = reconstruct_view_pos(s_uv, occ_depth);
        if (occ_pos.z >= s_pos.z + bias) {
            float dist_diff = abs(p_c.z - occ_pos.z);
            float t = clamp(radius / max(dist_diff, 1e-4), 0.0, 1.0);
            float range_check = t * t * (3.0 - 2.0 * t);
            occlusion += range_check;
        }
    }

    float normalized = (occlusion / float(kernel_size)) * intensity;
    float raw_vis = clamp(1.0 - normalized, 0.0, 1.0);
    float ao = clamp(pow(raw_vis, power), 0.0, 1.0);
    o_ao = vec4(ao, ao, ao, 1.0);
}
