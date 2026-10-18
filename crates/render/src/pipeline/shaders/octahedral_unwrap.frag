#version 450
// Unwraps a 6-face omnidirectional point-light depth strip into an octahedral shadow atlas tile.
//
// Uses `decode_octahedral` (Cigolle et al., JCGT 2014) to map each texel of the octahedral tile
// to a unit 3D direction `d`, samples the corresponding 90-degree perspective cube face, converts
// Reverse-Z depth to normalized Euclidean radial distance `length(pos - light_pos) / radius` in
// `[0, 1]`, and writes it to `gl_FragDepth`.
//
// Compiled with:
//   glslc -O octahedral_unwrap.frag -o octahedral_unwrap.frag.spv

layout(set = 0, binding = 0) uniform sampler2D u_cube_strip;

layout(push_constant) uniform OctUnwrapPush {
    // x = z_near, y = z_far (light radius), z = tile_offset_x, w = tile_offset_y
    vec4 near_far_offset;
    // x = inv_tile_width, y = inv_tile_height, z = 0, w = 0
    vec4 inv_tile_size;
} pc;

layout(location = 0) in vec2 v_uv;

vec3 decode_octahedral(vec2 uv) {
    vec2 f = uv * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        float ox = n.x;
        float oy = n.y;
        n.x = (1.0 - abs(oy)) * (step(0.0, ox) * 2.0 - 1.0);
        n.y = (1.0 - abs(ox)) * (step(0.0, oy) * 2.0 - 1.0);
    }
    return normalize(n);
}

void main() {
    vec2 local_uv = clamp((gl_FragCoord.xy - pc.near_far_offset.zw) * pc.inv_tile_size.xy, vec2(0.0), vec2(1.0));
    vec3 d = decode_octahedral(local_uv);

    vec3 ad = abs(d);
    int face = 0;
    float c = ad.x;
    float rx = 0.0;
    float ry = 0.0;

    if (ad.x >= max(ad.y, ad.z)) {
        c = ad.x;
        if (d.x >= 0.0) {
            // Face 0 (+X): forward=(+1,0,0), up=(0,+1,0), right=cross(forward,up)=(0,0,+1)
            face = 0;
            rx = d.z;
            ry = d.y;
        } else {
            // Face 1 (-X): forward=(-1,0,0), up=(0,+1,0), right=cross(forward,up)=(0,0,-1)
            face = 1;
            rx = -d.z;
            ry = d.y;
        }
    } else if (ad.y >= max(ad.x, ad.z)) {
        c = ad.y;
        if (d.y >= 0.0) {
            // Face 2 (+Y): forward=(0,+1,0), up=(0,0,-1), right=cross(forward,up)=(-1,0,0)
            face = 2;
            rx = -d.x;
            ry = -d.z;
        } else {
            // Face 3 (-Y): forward=(0,-1,0), up=(0,0,+1), right=cross(forward,up)=(-1,0,0)
            face = 3;
            rx = -d.x;
            ry = d.z;
        }
    } else {
        c = ad.z;
        if (d.z >= 0.0) {
            // Face 4 (+Z): forward=(0,0,+1), up=(0,+1,0), right=cross(forward,up)=(-1,0,0)
            face = 4;
            rx = -d.x;
            ry = d.y;
        } else {
            // Face 5 (-Z): forward=(0,0,-1), up=(0,+1,0), right=cross(forward,up)=(+1,0,0)
            face = 5;
            rx = d.x;
            ry = d.y;
        }
    }

    c = max(c, 1e-6);
    float face_u = clamp(0.5 * (rx / c + 1.0), 0.002, 0.998);
    float face_v = clamp(0.5 * (1.0 - ry / c), 0.002, 0.998);
    vec2 strip_uv = vec2((float(face) + face_u) / 6.0, face_v);

    float raw_depth = texture(u_cube_strip, strip_uv).r;
    if (raw_depth <= 1e-6) {
        // Clear depth 0.0 in Reverse-Z means no blocker in range -> radial depth = 1.0 (far).
        gl_FragDepth = 1.0;
    } else {
        float z_near = pc.near_far_offset.x;
        float z_far = pc.near_far_offset.y;
        float chebyshev_dist = (z_near * z_far) / (raw_depth * (z_far - z_near) + z_near);
        float radial_dist = chebyshev_dist / c;
        gl_FragDepth = clamp(radial_dist / z_far, 0.0, 1.0);
    }
}
