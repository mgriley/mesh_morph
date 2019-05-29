#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform Unifs {
  mat4 model;
  mat4 view;
  mat4 proj;

// BEGIN_USER_UNIFS
  // comps 1 min 0.0 max 4.0 speed 0.1 def 3.0
  vec4 render_mode;
  // comps 1 min 0.0 max 10.0 speed 0.1 def 5.0
  vec4 point_size;
  // comps 3 min 0.0 max 1.0 speed 0.1 def 1.0 0.0 0.0
  vec4 nor_display;
// END_USER_UNIFS
} unif;

layout(location = 0) in vec4 vs_pos;
layout(location = 1) in vec4 vs_vel;
layout(location = 2) in vec4 vs_neighbors;
layout(location = 3) in vec4 vs_data;
layout(location = 4) in vec4 vs_top_data;

layout(binding = 0, rgba32f) uniform readonly imageBuffer buf_pos;
layout(binding = 1, rgba32f) uniform readonly imageBuffer buf_vel;
layout(binding = 2, rgba32f) uniform readonly imageBuffer buf_neighbors;
layout(binding = 3, rgba32f) uniform readonly imageBuffer buf_data;
layout(binding = 4, rgba32f) uniform readonly imageBuffer buf_top_data;

layout(location = 0) out vec3 fs_nor;
layout(location = 1) out vec3 fs_col;

// Returns the normal as xyz, and length as w.
// If v is a zero-vec, returns length 0 and a norm of (1,0,0)
vec4 safe_norm(vec3 v) {
  float len = length(v);
  vec3 norm = len == 0.0 ? vec3(1.0,0.0,0.0) : v / len;
  return vec4(norm, len);
}

// Note: copied from morph.comp
vec3 node_normal(vec3 node_pos, vec4 node_neighbors) {
  vec3 nor = vec3(0.0,1.0,0.0);
  for (int i = 0; i < 4; ++i) {
    int i_a = int(node_neighbors[i]);
    int i_b = int(node_neighbors[(i + 1) % 4]);
    if (i_a != -1 && i_b != -1) {
      vec3 p_a = imageLoad(buf_pos, i_a).xyz;
      vec3 p_b = imageLoad(buf_pos, i_b).xyz;
      // TODO - why is this the 'up' direction, seems like the negative
      // sign should be unneccessary
      nor = safe_norm(-cross(p_a - node_pos, p_b - node_pos)).xyz;
      break;
    }
  }
  return nor;
}

// TODO - use this in morph.comp, too?
vec3 avg_node_normal(vec3 node_pos, vec4 node_neighbors) {
  vec3 avg_nor = vec3(0.0);
  int num_nors = 0;
  for (int i = 0; i < 4; ++i) {
    int i_a = int(node_neighbors[i]);
    int i_b = int(node_neighbors[(i + 1) % 4]);
    if (i_a != -1 && i_b != -1) {
      vec3 p_a = imageLoad(buf_pos, i_a).xyz;
      vec3 p_b = imageLoad(buf_pos, i_b).xyz;
      // TODO - why is this the 'up' direction, seems like the negative
      // sign should be unneccessary
      avg_nor += normalize(-cross(p_a - node_pos, p_b - node_pos));
      num_nors += 1;
    }
  }
  return avg_nor / num_nors;
}

void main() {
  vec3 col = vec3(0.0,1.0,0.0);
  vec3 nor = node_normal(vs_pos.xyz, vs_neighbors);

  if (int(unif.render_mode.x) == 1) {
    // show heat amount
    vec3 cold_col = vec3(0.0,0.0,1.0);
    vec3 hot_col = vec3(1.0,0.0,0.0);
    float mix_amt = clamp(vs_pos.w / 1.0f, 0.0, 1.0);
    col = mix(cold_col, hot_col, mix_amt);
  } else if (int(unif.render_mode.x) == 2) {
    // show heat generation
    float heat_gen_amt = clamp(vs_vel.w, 0.0, 1.0);
    col = vec3(heat_gen_amt, 0.0, 0.0);
  } else if (int(unif.render_mode.x) == 3) {
    // show heat sources
    float heat_gen_amt = vs_vel.w > 0.0 ? 1.0 : 0.0;
    col = vec3(heat_gen_amt, 0.0, 0.0);
  } else if (int(unif.render_mode.x) == 4) {
    // show normals
    col = unif.nor_display.xyz * nor;
  }

  fs_nor = nor;
  fs_col = col;
  gl_PointSize = unif.point_size.x;
  gl_Position = unif.proj * unif.view *
    unif.model * vec4(vs_pos.xyz, 1.0);
}

