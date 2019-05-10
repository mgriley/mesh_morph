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

struct AppState {
  vector<UserUnif> render_unifs;
  vector<UserUnif> compute_unifs;

  GLFWwindow* win = nullptr;
  bool framebuffer_resized = false;

  vector<uint16_t> indices;

  PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_utils;

  VkVertexInputBindingDescription binding_desc;
  array<VkVertexInputAttributeDescription, 3> attr_descs;

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

  VkDescriptorPool desc_pool;
  VkDescriptorSetLayout desc_set_layout;
  vector<VkDescriptorSet> desc_sets;

  VkRenderPass render_pass;
  VkPipelineLayout render_pipeline_layout;
  VkPipeline graphics_pipeline;
  vector<VkFramebuffer> swapchain_framebuffers;

  VkBuffer vert_buffer;
  VkDeviceMemory vert_buffer_mem;
  VkBuffer index_buffer;
  VkDeviceMemory index_buffer_mem;
  vector<VkBuffer> unif_buffers;
  vector<VkDeviceMemory> unif_buffers_mem;

  VkCommandPool cmd_pool;
  vector<VkCommandBuffer> cmd_buffers;
  
  vector<VkSemaphore> img_available_semas;
  vector<VkSemaphore> render_done_semas;
  vector<VkFence> in_flight_fences;

  VkImage texture_img;
  VkDeviceMemory texture_img_mem;
  VkImageView texture_img_view;
  VkSampler texture_sampler;

  VkImage depth_img;
  VkDeviceMemory depth_img_mem;
  VkImageView depth_img_view;

  size_t current_frame;
 
  AppState();
};


