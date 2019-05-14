#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform Unifs {
  mat4 model;
  mat4 view;
  mat4 proj;

// BEGIN_USER_UNIFS
  // comps 3 min 0.0 max 1.0 speed 0.01 def 1.0 0.0 0.0
  vec4 col;
  // comps 1 min 0.0 max 0.0 speed 0.1 def 0.5
  vec4 foo;
  // comps 2 min -1.0 max 1.0 speed 1.0 def 1.0
  vec4 bar;
// END_USER_UNIFS
} unif;


layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec4 in_vel;
layout(location = 2) in vec4 in_neighbors;
layout(location = 3) in vec4 in_data;

layout(location = 0) out vec3 frag_color;

// TODO - are these bindings correct?
/*
layout(binding = 4) uniform imageBuffer buf_pos;
layout(binding = 5) uniform imageBuffer buf_vel;
layout(binding = 6) uniform imageBuffer buf_neighbors;
layout(binding = 7) uniform imageBuffer buf_data;
*/

void main() {
  gl_PointSize = 5.0f;
  gl_Position = unif.proj * unif.view * unif.model *
    vec4(in_pos.xyz, 1.0);
  frag_color = unif.col.rgb;
}

