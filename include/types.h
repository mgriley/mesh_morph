#pragma once

#include "utils.h"
#include "vk_mem_alloc.h"

const int MAX_NUM_USER_UNIFS = 100;
// TODO - increase later
const uint32_t MAX_STORAGE_QUEUE_LEN = 1*1000; 

class AppState;

enum Attributes {
  ATTRIB_POS = 0,
  ATTRIB_VEL,
  ATTRIB_NEIGHBORS,
  ATTRIB_DATA,
  ATTRIB_TOP_DATA,

  ATTRIBUTES_COUNT
};

enum PipelineTypes {
  POINTS_PIPELINE,
  LINES_PIPELINE,
  TRIANGLES_PIPELINE,

  PIPELINES_COUNT
};

struct Camera {
  mat4 cam_to_world = mat4(1.0);

  Camera();
  void set_view(vec3 eye, vec3 target);
  vec3 pos() const;
  vec3 right() const;
  vec3 up() const;
  vec3 forward() const;
};

struct Controls {
  bool show_dev_console = true;
  int target_fps = 30;

  // rendering
  array<bool, PIPELINES_COUNT> pipeline_toggles =
    {true, true, false};

  // simulation
  bool log_input_nodes = false;
  bool log_output_nodes = false;
  bool log_input_compute_storage = false;
  bool log_output_compute_storage = false;
  bool log_point_indices = false;
  bool log_line_indices = false;
  bool log_triangle_indices = false;
  bool log_durations = false;
  int num_zygote_samples = 40;
  int inactive_node_count = 1000;
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

struct ComputeStorage {
  array<uint32_t, 2> step_counters = {0, 0};

  // for circular buffer queue
  array<uint32_t, 2> start_ptrs = {0, 0};
  array<uint32_t, 2> end_ptrs = {0, 0};
  array<uint32_t, MAX_STORAGE_QUEUE_LEN> queue_mem = {0};

  ComputeStorage();
};

struct RenderPushConstants {
  mat4 model = mat4(1.0);
  mat4 view = mat4(1.0);
  mat4 proj = mat4(1.0);
  array<vec4, MAX_NUM_USER_UNIFS> user_unif_vals;

  RenderPushConstants(mat4 model, mat4 view, mat4 proj,
      const vector<UserUnif>& user_unifs);
};

struct ComputePushConstants {
  uint32_t node_count;
  uint32_t inactive_node_count;
  uint32_t iter_num;
  uint32_t queue_len;
  array<vec4, MAX_NUM_USER_UNIFS> user_unif_vals;

  ComputePushConstants(uint32_t node_count, uint32_t inactive_node_count,
      uint32_t iter_num,
      uint32_t queue_len,
      const vector<UserUnif>& user_unifs);
};

struct MorphNode {
  vec4 pos = vec4(0.0);
  vec4 vel = vec4(0.0);
  // in order of: {right, upper, left, lower} wrt surface normal
  vec4 neighbors = vec4(0.0);
  vec4 data = vec4(0.0);
  vec4 top_data = vec4(0.0);

  MorphNode();
  MorphNode(vec4 pos, vec4 vel, vec4 neighbors,
      vec4 data, vec4 top_data);
};

struct MorphNodes {
  vector<vec4> pos_vec;
  vector<vec4> vel_vec;
  vector<vec4> neighbors_vec;
  vector<vec4> data_vec;
  vector<vec4> top_data_vec;

  MorphNodes(size_t num_nodes);
  MorphNodes(vector<MorphNode> const& nodes);
  MorphNode node_at(size_t i) const;

  array<void*, ATTRIBUTES_COUNT> data_ptrs();
};

string raw_node_str(MorphNode const& node);

// A single buffer in the double-buffered simulation
struct BufferState {
  array<VkBuffer, ATTRIBUTES_COUNT> vert_buffers;
  array<VmaAllocation, ATTRIBUTES_COUNT> vert_buffer_allocs;
  array<VkBufferView, ATTRIBUTES_COUNT> vert_buffer_views;

  VkDescriptorSet render_desc_set = VK_NULL_HANDLE;
  VkDescriptorSet compute_desc_set = VK_NULL_HANDLE;

  BufferState();
};

struct AppState {
  array<BufferState, 2> buffer_states;
  int result_buffer = 0;
  // number of vertices currently in the vertex buffers
  uint32_t node_count = 0;

  VmaAllocator allocator;

  VkBuffer compute_storage_buffer;
  VmaAllocation compute_storage_buffer_alloc;

  Camera cam;
  Controls controls;

  vector<UserUnif> render_unifs;
  vector<UserUnif> compute_unifs;

  GLFWwindow* win = nullptr;
  bool framebuffer_resized = false;

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
  array<VkPipeline, PIPELINES_COUNT> graphics_pipelines;

  VkPipelineLayout compute_pipeline_layout;
  VkPipeline compute_pipeline;

  vector<VkFramebuffer> swapchain_framebuffers;

  // contains the index data for rendering with each different
  // graphics pipeline
  array<VkBuffer, PIPELINES_COUNT> index_buffers;
  array<VmaAllocation, PIPELINES_COUNT> index_buffer_allocs;
  array<uint32_t, PIPELINES_COUNT> index_counts;

  VkCommandPool cmd_pool;
  vector<VkCommandBuffer> cmd_buffers;
  
  vector<VkSemaphore> img_available_semas;
  vector<VkSemaphore> render_done_semas;
  vector<VkFence> in_flight_fences;

  VkImage depth_img;
  VmaAllocation depth_img_alloc;
  VkImageView depth_img_view;

  size_t current_frame;
 
  AppState();
};


