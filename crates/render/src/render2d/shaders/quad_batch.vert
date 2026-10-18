#version 450
// Batched 2D quads (`QuadBatch2d`): positions in canvas units with a top-left origin, colours
// sRGB-encoded as authored.
//
// Compiled to `quad_batch.vert.spv` with:
//   glslc -O quad_batch.vert -o quad_batch.vert.spv
// Recompile and commit the SPIR-V whenever this file changes.

layout(push_constant) uniform Canvas {
    // 2 / canvas width, 2 / canvas height.
    vec2 inv_half_size;
    // 1.0 when the target is sRGB-encoded: colours are decoded to linear so that the hardware
    // encoding reproduces the authored values.
    float decode_srgb;
} canvas;

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

vec3 srgb_to_linear(vec3 c) {
    vec3 low = c / 12.92;
    vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, lessThanEqual(c, vec3(0.04045)));
}

void main() {
    // Canvas y grows downwards; RHI clip space is +Y up.
    vec2 ndc = vec2(position.x * canvas.inv_half_size.x - 1.0,
                    1.0 - position.y * canvas.inv_half_size.y);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vec3 rgb = canvas.decode_srgb > 0.5 ? srgb_to_linear(color.rgb) : color.rgb;
    v_color = vec4(rgb, color.a);
    v_uv = uv;
}
