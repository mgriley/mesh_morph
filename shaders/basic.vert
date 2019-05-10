#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform UserUnifs {
  // comps 1 min 0.0 max 0.0 speed 1.0 def 0.5
  vec4 foo;
  // comps 2 min -1.0 max 1.0 speed 1.0 def 1.0
  vec4 bar;
} uu;

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
} ubo;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec2 in_tex_coord;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 frag_tex_coord;

void main() {
  gl_Position = ubo.proj * ubo.view * ubo.model *
    vec4(in_pos, 1.0);
  frag_color = in_color;
  frag_tex_coord = in_tex_coord;
}
