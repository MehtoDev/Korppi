#version 450
layout(push_constant) uniform PC { layout(offset = 16) vec4 color; } pc;
layout(location = 0) out vec4 o;
void main() { o = pc.color; }
