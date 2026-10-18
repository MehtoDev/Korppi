#version 450
// Depth-only vertex shader used by the 1x Reverse-Z depth prepass, directional CSM shadow cascades,
// spotlight perspective shadow maps, and omnidirectional point-light cube-strip shadow passes.
//
// Compiled with:
//   glslc -O depth_prepass.vert -o depth_prepass.vert.spv

layout(push_constant) uniform DepthPush {
    mat4 mvp;
} pc;

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

void main() {
    gl_Position = pc.mvp * vec4(a_pos, 1.0);
}
