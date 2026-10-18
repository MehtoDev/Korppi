#version 450
// Final HDR-to-LDR composite and tonemapping pass matching `evaluate_tonemapping` (`RND-01`, `RND-15`).
//
// Supports AgX (Troy Sobotka log2 range [-12.47393, +4.026069] EV + 6th-order polynomial),
// fitted ACES (Stephen Hill / Krzysztof Narkowicz), extended luminance Reinhard, and Uncharted 2 Filmic.
//
// Compiled with:
//   glslc -O tonemap.frag -o tonemap.frag.spv

layout(set = 0, binding = 0) uniform sampler2D u_hdr;
layout(set = 0, binding = 1) uniform sampler2D u_bloom;

layout(push_constant) uniform TonemapPush {
    // x = exposure, y = gamma, z = white_point, w = bloom_intensity
    vec4 params;
    // x = operator (0 = AgX, 1 = Aces, 2 = Reinhard, 3 = Filmic), y = has_bloom, z = srgb_target, w = 0
    uvec4 flags;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

vec3 reinhard_luminance(vec3 color, float white_point) {
    vec3 c = max(color, vec3(0.0));
    float l = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    if (l <= 1e-6) {
        return vec3(0.0);
    }
    float w2 = max(white_point * white_point, 1e-4);
    float l_out = (l * (1.0 + l / w2)) / (1.0 + l);
    return clamp(c * (l_out / l), vec3(0.0), vec3(1.0));
}

vec3 aces_fitted(vec3 color) {
    vec3 v = max(color, vec3(0.0));
    vec3 num = v * (2.51 * v + vec3(0.03));
    vec3 den = v * (2.43 * v + vec3(0.59)) + vec3(0.14);
    return clamp(num / den, vec3(0.0), vec3(1.0));
}

vec3 hable_part(vec3 x) {
    float a = 0.15;
    float b = 0.50;
    float c = 0.10;
    float d = 0.20;
    float e = 0.02;
    float f = 0.30;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - vec3(e / f);
}

vec3 filmic_hable(vec3 color) {
    vec3 v = max(color, vec3(0.0)) * 2.0;
    vec3 curr = hable_part(v);
    float white_scale = 1.0 / hable_part(vec3(11.2)).x;
    return clamp(curr * white_scale, vec3(0.0), vec3(1.0));
}

float agx_contrast(float x) {
    float x2 = x * x;
    float x4 = x2 * x2;
    return clamp(
        15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232,
        0.0,
        1.0
    );
}

vec3 agx_tonemap(vec3 color) {
    vec3 v = max(color, vec3(0.0));
    float r = 0.84247906 * v.r + 0.0784336 * v.g + 0.07922375 * v.b;
    float g = 0.04232824 * v.r + 0.87846863 * v.g + 0.07916613 * v.b;
    float b = 0.04237565 * v.r + 0.0784336 * v.g + 0.879143 * v.b;
    vec3 agx_in = max(vec3(r, g, b), vec3(1e-10));

    float min_ev = -12.47393;
    float max_ev = 4.026069;
    vec3 log_v = clamp((log2(agx_in) - vec3(min_ev)) / (max_ev - min_ev), vec3(0.0), vec3(1.0));

    float cx = agx_contrast(log_v.x);
    float cy = agx_contrast(log_v.y);
    float cz = agx_contrast(log_v.z);

    float out_r = 1.196879 * cx - 0.09802088 * cy - 0.09902974 * cz;
    float out_g = -0.05289685 * cx + 1.1519031 * cy - 0.09896118 * cz;
    float out_b = -0.05297164 * cx - 0.09804345 * cy + 1.1510737 * cz;
    return clamp(vec3(out_r, out_g, out_b), vec3(0.0), vec3(1.0));
}

vec3 srgb_to_linear(vec3 c) {
    vec3 low = c / 12.92;
    vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, lessThanEqual(c, vec3(0.04045)));
}

void main() {
    vec3 hdr = texture(u_hdr, v_uv).rgb;
    if (pc.flags.y != 0u) {
        vec3 bloom = texture(u_bloom, v_uv).rgb;
        hdr += bloom * max(pc.params.w, 0.0);
    }

    vec3 exposed = hdr * max(pc.params.x, 0.0);
    uint op = pc.flags.x;
    vec3 display_rgb;

    if (op == 0u) {
        display_rgb = agx_tonemap(exposed);
    } else {
        vec3 mapped;
        if (op == 1u) {
            mapped = aces_fitted(exposed);
        } else if (op == 2u) {
            mapped = reinhard_luminance(exposed, pc.params.z);
        } else {
            mapped = filmic_hable(exposed);
        }
        float inv_gamma = 1.0 / max(pc.params.y, 0.001);
        display_rgb = clamp(pow(max(mapped, vec3(0.0)), vec3(inv_gamma)), vec3(0.0), vec3(1.0));
    }

    if (pc.flags.z != 0u) {
        display_rgb = srgb_to_linear(display_rgb);
    }

    o_color = vec4(display_rgb, 1.0);
}
