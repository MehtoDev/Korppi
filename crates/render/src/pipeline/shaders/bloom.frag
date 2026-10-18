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

vec3 soft_threshold(vec3 color, float threshold, float soft_knee) {
    vec3 c = max(color, vec3(0.0));
    float brightness = max(c.r, max(c.g, c.b));
    if (brightness <= 1e-6) {
        return vec3(0.0);
    }
    float knee = threshold * soft_knee + 1e-5;
    float soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    float quadratic = (soft * soft) / (4.0 * knee);
    float contribution = max(brightness - threshold, quadratic) / brightness;
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

        vec3 filtered = g * 0.125
            + (d + e + i + j) * 0.125
            + (b + f + h + l) * 0.0625
            + (a + c + k + m) * 0.03125;

        if (pc.threshold_params.x > 0.0) {
            filtered = soft_threshold(filtered, pc.threshold_params.x, pc.threshold_params.y);
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
