#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fs_nor;
layout(location = 1) in vec3 fs_col;

layout(location = 0) out vec4 out_color;

void main() {
  vec3 world_light_dir = normalize(vec3(1.0,1.0,1.0));
  vec3 world_nor = normalize(fs_nor);
  float df = clamp(dot(world_nor, world_light_dir), 0.0, 1.0);
  vec3 col = (df + 0.2) * fs_col;

  // TODO - debug color?
  //col = debug_render ? fs_col : col;

  out_color = vec4(col, 1.0);  
}

