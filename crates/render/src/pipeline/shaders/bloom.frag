#version 450
// Progressive Jorge Jimenez 13-tap downsample (with soft-knee threshold on mip 0) and 9-tap tent
// upsample filter (`RND-01`, `RND-15`).
//
// Compiled with:
//   glslc -O bloom.frag -o bloom.frag.spv

layout(set = 0, binding = 0) uniform sampler2D u_src;

layout(push_constant) uniform BloomPush {
    // x = texel_x, y = texel_y, z = filter_radius, w = mode (0.0 = downsample+threshold, 1.0 = upsample)
    vec4 texel_radius_mode;
    // x = threshold, y = soft_knee, z = 0, w = 0
    vec4 threshold_params;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

float rec709_luma(vec3 c) {
    return dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

float karis_weight(vec3 c) {
    return 1.0 / (1.0 + rec709_luma(c));
}

vec3 soft_threshold(vec3 color, float threshold, float soft_knee) {
    vec3 c = max(color, vec3(0.0));
    float brightness = rec709_luma(c);
    if (brightness <= 1e-6) {
        return vec3(0.0);
    }
    float knee = threshold * soft_knee + 1e-4;
    float soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    float quadratic = 0.25 * (soft * soft) / knee;
    float contribution = max(brightness - threshold, quadratic) / max(brightness, 1e-4);
    return c * max(contribution, 0.0);
}

void main() {
    vec2 ts = pc.texel_radius_mode.xy * max(pc.texel_radius_mode.z, 1.0);

    if (pc.texel_radius_mode.w < 0.5) {
        // 13-tap Jimenez downsample with reference weights:
        // center = 0.125, inner 4 = 0.125, edges 4 = 0.0625, corners 4 = 0.03125 (sum = 1.0).
        vec3 a = texture(u_src, v_uv + vec2(-2.0, -2.0) * ts).rgb;
        vec3 b = texture(u_src, v_uv + vec2( 0.0, -2.0) * ts).rgb;
        vec3 c = texture(u_src, v_uv + vec2( 2.0, -2.0) * ts).rgb;
        vec3 d = texture(u_src, v_uv + vec2(-1.0, -1.0) * ts).rgb;
        vec3 e = texture(u_src, v_uv + vec2( 1.0, -1.0) * ts).rgb;
        vec3 f = texture(u_src, v_uv + vec2(-2.0,  0.0) * ts).rgb;
        vec3 g = texture(u_src, v_uv).rgb;
        vec3 h = texture(u_src, v_uv + vec2( 2.0,  0.0) * ts).rgb;
        vec3 i = texture(u_src, v_uv + vec2(-1.0,  1.0) * ts).rgb;
        vec3 j = texture(u_src, v_uv + vec2( 1.0,  1.0) * ts).rgb;
        vec3 k = texture(u_src, v_uv + vec2(-2.0,  2.0) * ts).rgb;
        vec3 l = texture(u_src, v_uv + vec2( 0.0,  2.0) * ts).rgb;
        vec3 m = texture(u_src, v_uv + vec2( 2.0,  2.0) * ts).rgb;

        vec3 filtered;
        if (pc.threshold_params.x > 0.0) {
            // First downsample pass: Karis-weighted average for firefly suppression (`RND-20`)
            float wg = 0.125 * karis_weight(g);
            float wd = 0.125 * karis_weight(d);
            float we = 0.125 * karis_weight(e);
            float wi = 0.125 * karis_weight(i);
            float wj = 0.125 * karis_weight(j);
            float wb = 0.0625 * karis_weight(b);
            float wf = 0.0625 * karis_weight(f);
            float wh = 0.0625 * karis_weight(h);
            float wl = 0.0625 * karis_weight(l);
            float wa = 0.03125 * karis_weight(a);
            float wc = 0.03125 * karis_weight(c);
            float wk = 0.03125 * karis_weight(k);
            float wm = 0.03125 * karis_weight(m);
            float w_sum = max(wg + wd + we + wi + wj + wb + wf + wh + wl + wa + wc + wk + wm, 1e-6);
            filtered = (
                g * wg + d * wd + e * we + i * wi + j * wj
                + b * wb + f * wf + h * wh + l * wl
                + a * wa + c * wc + k * wk + m * wm
            ) / w_sum;
            filtered = soft_threshold(filtered, pc.threshold_params.x, pc.threshold_params.y);
        } else {
            filtered = g * 0.125
                + (d + e + i + j) * 0.125
                + (b + f + h + l) * 0.0625
                + (a + c + k + m) * 0.03125;
        }
        o_color = vec4(filtered, 1.0);
    } else {
        // 9-tap 3x3 tent upsample filter:
        // center = 4/16, edges = 2/16, corners = 1/16
        vec3 tl = texture(u_src, v_uv + vec2(-1.0, -1.0) * ts).rgb;
        vec3 tc = texture(u_src, v_uv + vec2( 0.0, -1.0) * ts).rgb;
        vec3 tr = texture(u_src, v_uv + vec2( 1.0, -1.0) * ts).rgb;
        vec3 ml = texture(u_src, v_uv + vec2(-1.0,  0.0) * ts).rgb;
        vec3 mc = texture(u_src, v_uv).rgb;
        vec3 mr = texture(u_src, v_uv + vec2( 1.0,  0.0) * ts).rgb;
        vec3 bl = texture(u_src, v_uv + vec2(-1.0,  1.0) * ts).rgb;
        vec3 bc = texture(u_src, v_uv + vec2( 0.0,  1.0) * ts).rgb;
        vec3 br = texture(u_src, v_uv + vec2( 1.0,  1.0) * ts).rgb;

        vec3 tent = (tl + tr + bl + br) * (1.0 / 16.0)
            + (tc + ml + mr + bc) * (2.0 / 16.0)
            + mc * (4.0 / 16.0);
        o_color = vec4(tent, 1.0);
    }
}
