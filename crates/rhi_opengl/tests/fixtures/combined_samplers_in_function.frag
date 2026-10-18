#version 450
layout(set = 1, binding = 2) uniform sampler2D a;
layout(set = 1, binding = 3) uniform sampler2D b;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
vec4 helper(vec2 p) { return texture(a, p) + texture(b, p * 2.0); }
void main() { o = helper(uv) + textureLod(a, uv, 0.0); }
