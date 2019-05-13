#pragma once

#include "utils.h"


struct Vertex {
  vec3 pos;
  vec3 color;
  vec2 tex_coord;
};

struct UserUnif {
  string name;
  int num_comps = 4;
  vec4 default_val = vec4(0.0);
  vec4 current_val = vec4(0.0);
  float min_val = 0;
  float max_val = 1.0;
  float drag_speed = 1.0;

  UserUnif(string name, int num_comps,
      vec4 def_val, float min_val, float max_val,
      float drag_speed);
};

struct UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
};

enum Attributes {
  ATTR_POS = 0,
  ATTRI_VEL,
  ATTR_NEIGHBORS,
  ATTRI_DATA,

  ATTRIBUTES_COUNT
};

// A single buffer in the double-buffered simulation
struct BufferState {
  array<VkBuffer, ATTRIBUTES_COUNT> vert_buffers;
  array<VkDeviceMemory, ATTRIBUTES_COUNT> vert_buffer_mems;
  array<VkBufferView, ATTRIBUTES_COUNT> vert_buffer_views;

  VkDescriptorSet render_desc_set;
  VkDescriptorSet compute_desc_set;

  BufferState();
};

struct AppState {
  array<BufferState, 2> buffer_states;
  vector<UserUnif> render_unifs;
  vector<UserUnif> compute_unifs;

  GLFWwindow* win = nullptr;
  bool framebuffer_resized = false;

  vector<uint16_t> indices;

  PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_utils;
  
  VkInstance inst;
  VkDebugUtilsMessengerEXT debug_messenger;
  VkSurfaceKHR surface;
  VkPhysicalDevice phys_device;
  VkDevice device;
  uint32_t target_family_index;
  VkQueue queue;

  VkSurfaceCapabilitiesKHR surface_caps;
  VkSurfaceFormatKHR target_format;
  VkPresentModeKHR target_present_mode;
  VkExtent2D target_extent;
  uint32_t target_image_count;

  VkSwapchainKHR swapchain;
  vector<VkImage> swapchain_images;
  vector<VkImageView> swapchain_img_views;

  vector<VkVertexInputBindingDescription> vert_binding_descs;
  vector<VkVertexInputAttributeDescription> vert_attr_descs;

  VkDescriptorPool desc_pool;
  VkDescriptorSetLayout render_desc_set_layout;
  VkDescriptorSetLayout compute_desc_set_layout;

  VkRenderPass render_pass;
  VkPipelineLayout render_pipeline_layout;
  VkPipeline graphics_pipeline;

  VkPipelineLayout compute_pipeline_layout;
  VkPipeline compute_pipeline;

  vector<VkFramebuffer> swapchain_framebuffers;

  VkBuffer index_buffer;
  VkDeviceMemory index_buffer_mem;

  VkCommandPool cmd_pool;
  vector<VkCommandBuffer> cmd_buffers;
  
  vector<VkSemaphore> img_available_semas;
  vector<VkSemaphore> render_done_semas;
  vector<VkFence> in_flight_fences;

  VkImage depth_img;
  VkDeviceMemory depth_img_mem;
  VkImageView depth_img_view;

  size_t current_frame;
 
  AppState();
};


