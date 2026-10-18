#version 450
// Clustered Forward PBR vertex shader (`RND-01`).
//
// Compiled with:
//   glslc -O forward_pbr.vert -o forward_pbr.vert.spv

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 ambient_light;
    vec4 dir_light_dir;
    vec4 dir_light_radiance;
    uvec4 cluster_dims;
    vec4 cluster_depth_params;
    vec4 viewport_and_shadow_counts;
    mat4 csm_view_proj[4];
    vec4 csm_atlas_uv[4];
    vec4 csm_split_depths;
    mat4 spot_shadow_view_proj[4];
    vec4 spot_shadow_atlas_uv[4];
    uvec4 spot_shadow_light_indices;
    vec4 point_shadow_pos_radius[4];
    vec4 point_shadow_atlas_uv[4];
    uvec4 point_shadow_light_indices;
} scene;

layout(push_constant) uniform DrawPush {
    mat4 model;
    // rgb = albedo, w = perceptual_roughness
    vec4 albedo_roughness;
    // rgb = emissive, w = metallic
    vec4 emissive_metallic;
    // x = reflectance, y = clear_coat_roughness, z = clear_coat_weight, w = has_clear_coat
    vec4 reflectance_clearcoat;
    // rgb = sheen_color, w = sheen_roughness (0 if disabled)
    vec4 sheen;
} pc;

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_world_normal;
layout(location = 2) out vec3 v_view_pos;
layout(location = 3) out vec3 v_view_normal;
layout(location = 4) out vec2 v_uv;

void main() {
    vec4 world_pos = pc.model * vec4(a_pos, 1.0);
    mat3 normal_mat = transpose(inverse(mat3(pc.model)));
    vec3 world_normal = normalize(normal_mat * a_normal);
    vec4 view_pos = scene.view * world_pos;
    vec3 view_normal = normalize(mat3(scene.view) * world_normal);

    v_world_pos = world_pos.xyz;
    v_world_normal = world_normal;
    v_view_pos = view_pos.xyz;
    v_view_normal = view_normal;
    v_uv = a_uv;
    gl_Position = scene.view_proj * world_pos;
}
