#version 450
// Fullscreen triangle vertex shader covering [-1, 1]^2 in +Y-up NDC with top-left UV origin (0, 0).
//
// Compiled with:
//   glslc -O fullscreen.vert -o fullscreen.vert.spv

layout(location = 0) out vec2 v_uv;

const vec2 POSITIONS[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    vec2 ndc = POSITIONS[gl_VertexIndex];
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = vec2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y));
}
