#pragma once

#include "utils.h"

const int MAX_NUM_USER_UNIFS = 100;

struct Camera {
  mat4 cam_to_world = mat4(1.0);

  Camera();
  void set_view(vec3 eye, vec3 target);
  vec3 pos() const;
  vec3 right() const;
  vec3 up() const;
  vec3 forward() const;
};

// User controls 
struct Controls {
  int target_fps = 30;

  // rendering
  bool render_faces = false;
  bool render_points = true;
  bool render_wireframe = true;

  // simulation
  bool log_input_nodes = false;
  bool log_output_nodes = false;
  bool log_render_data = false;
  bool log_durations = false;
  int num_zygote_samples = 100;
  // for the simulation/animation pane
  int num_iters = 0;
  bool animating_sim = true;
  bool loop_at_end = false;
  int start_iter_num = 0;
  int end_iter_num = 10*1000*1000;
  int delta_iters = 0;

  bool cam_spherical_mode = true;

  Controls();
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

struct RenderPushConstants {
  mat4 model;
  mat4 view;
  mat4 proj;

  array<vec4, MAX_NUM_USER_UNIFS> user_unif_vals;

  RenderPushConstants(mat4 model, mat4 view, mat4 view,
      const vector<UserUnif>& user_unifs);
};

struct ComputePushConstants {

  array<vec4, MAX_NUM_USER_UNIFS> user_unif_vals;

  ComputePushConstants();
};

enum Attributes {
  ATTRIB_POS = 0,
  ATTRIB_VEL,
  ATTRIB_NEIGHBORS,
  ATTRIB_DATA,

  ATTRIBUTES_COUNT
};

struct MorphNode {
  vec4 pos = vec4(0.0);
  vec4 vel = vec4(0.0);
  // in order of: {right, upper, left, lower} wrt surface normal
  vec4 neighbors = vec4(0.0);
  vec4 data = vec4(0.0);

  MorphNode();
  MorphNode(vec4 pos, vec4 vel, vec4 neighbors, vec4 data);
};

struct MorphNodes {
  vector<vec4> pos_vec;
  vector<vec4> vel_vec;
  vector<vec4> neighbors_vec;
  vector<vec4> data_vec;

  MorphNodes(size_t num_nodes);
  MorphNodes(vector<MorphNode> const& nodes);
  MorphNode node_at(size_t i) const;
};

string raw_node_str(MorphNode const& node);

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
  int result_buffer = 0;
  uint32_t node_count = 0;

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


