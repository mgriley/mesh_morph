#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform sampler2D tex_sampler;

void main() {
  //out_color = texture(tex_sampler, frag_tex_coord);
  //out_color = vec4(frag_tex_coord, 0.0, 1.0);
  out_color = vec4(frag_color, 1.0);
}
