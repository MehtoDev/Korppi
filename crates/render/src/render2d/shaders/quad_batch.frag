#version 450
// Textured and solid-colour fragment stage for batched 2D quads (see `quad_batch.vert`).
//
// Compiled to `quad_batch.frag.spv` with:
//   glslc -O quad_batch.frag -o quad_batch.frag.spv

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(location = 0) out vec4 o_color;

void main() {
    o_color = v_color * texture(u_texture, v_uv);
}

