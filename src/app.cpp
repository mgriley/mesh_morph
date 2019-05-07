#include "app.h"
#include "utils.h"
/*
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
*/

#include <chrono>
#include <thread>
#include <utility>
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace glm;

const int max_frames_in_flight = 2;

struct Vertex {
  vec3 pos;
  vec3 color;
  vec2 tex_coord;
};

struct UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
};

struct AppState {
  GLFWwindow* win = nullptr;

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

  VkRenderPass render_pass;
  VkDescriptorSetLayout desc_set_layout;
  VkPipelineLayout pipeline_layout;
  VkPipeline graphics_pipeline;
  vector<VkFramebuffer> swapchain_framebuffers;

  VkCommandPool cmd_pool;

  VkBuffer vert_buffer;
  VkDeviceMemory vert_buffer_mem;
  VkBuffer index_buffer;
  VkDeviceMemory index_buffer_mem;
  vector<VkBuffer> unif_buffers;
  vector<VkDeviceMemory> unif_buffers_mem;

  VkDescriptorPool desc_pool;
  vector<VkDescriptorSet> desc_sets;

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

AppState::AppState()
{
}

vector<char> read_file(const string& filename) {
  ifstream file(filename, ios::ate | ios::binary);
  size_t file_size = (size_t) file.tellg();
  vector<char> buffer(file_size);
  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();
  return buffer;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  VkDebugUtilsMessageTypeFlagsEXT messageType,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  void* pUserData) {

  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    printf("validation layer: %s\n", pCallbackData->pMessage);
  }

  return VK_FALSE;
}

VkShaderModule create_shader_module(VkDevice& device, const vector<char>& code) {
  VkShaderModuleCreateInfo create_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = code.size(),
    .pCode = reinterpret_cast<const uint32_t*>(code.data())
  };
  VkShaderModule module;
  VkResult res = vkCreateShaderModule(device, &create_info, nullptr, &module);
  assert(res == VK_SUCCESS);
  return module;
}

uint32_t find_mem_type_index(VkPhysicalDevice& phys_device,
    uint32_t type_filter,
    VkMemoryPropertyFlags target_mem_flags) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);

  uint32_t mem_type_index = 0;
  bool found_mem_type = false;
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    bool mem_type_supported = type_filter & (1 << i);
    bool has_target_props = (mem_props.memoryTypes[i].propertyFlags &
      target_mem_flags) == target_mem_flags;
    if (mem_type_supported && has_target_props) {
      mem_type_index = i;
      found_mem_type = true;
      break;
    }
  }
  assert(found_mem_type);
  return mem_type_index;
}

VkCommandBuffer begin_single_time_commands(AppState& state) {
  VkCommandBufferAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandPool = state.cmd_pool,
    .commandBufferCount = 1
  };
  VkCommandBuffer tmp_cmd_buffer;
  vkAllocateCommandBuffers(state.device, &alloc_info, &tmp_cmd_buffer);

  VkCommandBufferBeginInfo begin_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  };
  vkBeginCommandBuffer(tmp_cmd_buffer, &begin_info);

  return tmp_cmd_buffer;
}

void end_single_time_commands(AppState& state, VkCommandBuffer cmd_buffer) {
  vkEndCommandBuffer(cmd_buffer);

  VkSubmitInfo submit_info = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmd_buffer
  };
  vkQueueSubmit(state.queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(state.queue);

  vkFreeCommandBuffers(state.device, state.cmd_pool, 1, &cmd_buffer);
}



void copy_buffer(
    AppState& state,
    VkBuffer src_buffer, VkBuffer dst_buffer,
    VkDeviceSize buffer_size) {
  VkCommandBuffer tmp_cmd_buffer = begin_single_time_commands(state);

  VkBufferCopy copy_region = {
    .srcOffset = 0,
    .dstOffset = 0,
    .size = buffer_size
  };
  vkCmdCopyBuffer(tmp_cmd_buffer, src_buffer, dst_buffer,
      1, &copy_region);
  
  end_single_time_commands(state, tmp_cmd_buffer);
}

void create_buffer(
    VkDevice device,
    VkPhysicalDevice phys_device,
    VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags props, VkBuffer& buffer, VkDeviceMemory& buffer_mem) {
  VkBufferCreateInfo buffer_info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = size,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE
  };
  VkResult res = vkCreateBuffer(device, &buffer_info, nullptr, &buffer);
  assert(res == VK_SUCCESS);

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

  uint32_t mem_type_index = find_mem_type_index(
      phys_device, mem_reqs.memoryTypeBits, props);
  VkMemoryAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = mem_reqs.size,
    .memoryTypeIndex = mem_type_index
  };
  res = vkAllocateMemory(device, &alloc_info, nullptr, &buffer_mem);
  assert(res == VK_SUCCESS);

  vkBindBufferMemory(device, buffer, buffer_mem, 0);
}

void create_index_buffer(
    AppState& state,
    vector<uint16_t>& indices,
    VkBuffer& index_buffer, VkDeviceMemory& index_buffer_mem) {
  VkDeviceSize buffer_size = sizeof(indices[0]) * indices.size();

  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_mem;
  create_buffer(state.device, state.phys_device, buffer_size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      staging_buffer, staging_buffer_mem);

  void* data;
  vkMapMemory(state.device, staging_buffer_mem, 0, buffer_size, 0, &data);
  memcpy(data, indices.data(), (size_t) buffer_size);
  vkUnmapMemory(state.device, staging_buffer_mem);

  create_buffer(state.device, state.phys_device, buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        index_buffer, index_buffer_mem);

  copy_buffer(state,
      staging_buffer, index_buffer, buffer_size);

  vkDestroyBuffer(state.device, staging_buffer, nullptr);
  vkFreeMemory(state.device, staging_buffer_mem, nullptr);
}

VkFormat find_supported_format(VkPhysicalDevice& phys_device,
    const vector<VkFormat>& candidates, VkImageTiling tiling,
    VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(phys_device, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
        (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }
  throw std::runtime_error("could not find supported format");
}

VkFormat find_depth_format(VkPhysicalDevice& phys_device) {
  return find_supported_format(phys_device,
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL,
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

bool has_stencil_component(VkFormat format) {
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
    format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void enumerate_instance_extensions() {
  // enumerate instance extensions
  uint32_t num_exts = 0;
  VkResult res = vkEnumerateInstanceExtensionProperties(nullptr, &num_exts, nullptr);
  vector<VkExtensionProperties> ext_props{num_exts};
  res = vkEnumerateInstanceExtensionProperties(nullptr,
      &num_exts, ext_props.data());
  printf("exts:\n");
  for (int i = 0; i < num_exts; ++i) {
    printf("%s\n", ext_props[i].extensionName);
  }
  printf("\n");
}

void enumerate_instance_layers() {
  // enumerate layers
  uint32_t num_layers = 0;
  VkResult res = vkEnumerateInstanceLayerProperties(
      &num_layers, nullptr);
  vector<VkLayerProperties> layer_props{num_layers};
  res = vkEnumerateInstanceLayerProperties(
      &num_layers, layer_props.data());
  printf("layers:\n");
  for (int i = 0; i < num_layers; ++i) {
    printf("%s\n", layer_props[i].layerName);
  }
  printf("\n");
}

void setup_vertex_attr_desc(AppState& state) {
  state.binding_desc = {
    .binding = 0,
    .stride = sizeof(Vertex),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
  };
  state.attr_descs[0] = {
    .binding = 0,
    .location = 0,
    .format = VK_FORMAT_R32G32B32_SFLOAT,
    .offset = offsetof(Vertex, pos)
  };
  state.attr_descs[1] = {
    .binding = 0,
    .location = 1,
    .format = VK_FORMAT_R32G32B32_SFLOAT,
    .offset = offsetof(Vertex, color)
  };
  state.attr_descs[2] = {
    .binding = 0,
    .location = 2,
    .format = VK_FORMAT_R32G32_SFLOAT,
    .offset = offsetof(Vertex, tex_coord)
  };
}

void setup_instance(AppState& state) {
  enumerate_instance_extensions();
  enumerate_instance_layers();
  
  // gather extensions
  uint32_t glfw_ext_count = 0;
  const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
  vector<const char*> ext_names;
  ext_names.insert(ext_names.end(), glfw_exts, glfw_exts + glfw_ext_count);
  ext_names.push_back("VK_EXT_debug_utils");

  // gather layers
  vector<const char*> layer_names = {
    "VK_LAYER_KHRONOS_validation"
  };

  // setup instance
  VkApplicationInfo app_info = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pNext = nullptr,
    .pApplicationName = "my_vulkan_app",
    .applicationVersion = 1,
    .pEngineName = "my_vulkan_app",
    .engineVersion = 1,
    .apiVersion = VK_API_VERSION_1_0
  };
  VkInstanceCreateInfo inst_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .pApplicationInfo = &app_info,
    .enabledExtensionCount = static_cast<uint32_t>(ext_names.size()),
    .ppEnabledExtensionNames = ext_names.data(),
    .enabledLayerCount = static_cast<uint32_t>(layer_names.size()),
    .ppEnabledLayerNames = layer_names.data()
  };
  VkResult res = vkCreateInstance(&inst_info, nullptr, &state.inst);
  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
    printf("cant find a compatible vulkan ICD\n");
    exit(1);
  } else if (res) {
    printf("Error occurred on create instance\n");
    exit(1);
  }
}

void setup_debug_callback(AppState& state) {
  // setup debug callback
  VkDebugUtilsMessengerCreateInfoEXT debug_utils_info = {
    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    .pfnUserCallback = vulkan_debug_callback,
    .pUserData = nullptr
  };
  auto create_debug_utils = (PFN_vkCreateDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(state.inst, "vkCreateDebugUtilsMessengerEXT");
  state.destroy_debug_utils = (PFN_vkDestroyDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(state.inst, "vkDestroyDebugUtilsMessengerEXT");
  assert(create_debug_utils);
  assert(state.destroy_debug_utils);
  VkResult res = create_debug_utils(state.inst, &debug_utils_info,
      nullptr, &state.debug_messenger);
  assert(res == VK_SUCCESS);
}

void setup_surface(AppState& state) {
  // init surface
  VkResult res = glfwCreateWindowSurface(state.inst, state.win, nullptr,
      &state.surface);
  assert(res == VK_SUCCESS);
}

void setup_physical_device(AppState& state) {
  // retrieve physical device
  // assume the first GPU will do
  uint32_t device_count = 1;
  VkResult res = vkEnumeratePhysicalDevices(state.inst, &device_count,
      &state.phys_device);
  assert(!res && device_count == 1);
}

void setup_logical_device(AppState& state) {
  // find a suitable queue family
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(state.phys_device, &queue_family_count,
      nullptr);
  assert(queue_family_count > 0);
  vector<VkQueueFamilyProperties> queue_fam_props{queue_family_count};
  vkGetPhysicalDeviceQueueFamilyProperties(state.phys_device, &queue_family_count,
      queue_fam_props.data());
  assert(queue_family_count > 0);

  bool found_index = false;
  printf("queue families:\n");
  for (int i = 0; i < queue_family_count; ++i) {
    auto& q_fam = queue_fam_props[i];
    bool supports_graphics = q_fam.queueFlags & VK_QUEUE_GRAPHICS_BIT;
    bool supports_compute = q_fam.queueFlags & VK_QUEUE_COMPUTE_BIT;
    VkBool32 supports_present = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(state.phys_device, i, state.surface, &supports_present);
    printf("G: %i, C: %i, P: %d, count: %d\n", supports_graphics ? 1 : 0,
        supports_compute ? 1 : 0, supports_present, q_fam.queueCount);

    if (supports_graphics && supports_compute && supports_present) {
      state.target_family_index = i;
      found_index = true;
    }
  }
  printf("\n");
  assert(found_index);

  // init logical device

  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .pNext = nullptr,
    .queueFamilyIndex = state.target_family_index,
    .queueCount = 1,
    .pQueuePriorities = &queue_priority
  };
  vector<const char*> device_ext_names = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };
  VkDeviceCreateInfo device_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = nullptr,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queue_info,
    .enabledExtensionCount = static_cast<uint32_t>(device_ext_names.size()),
    .ppEnabledExtensionNames = device_ext_names.data(),
    .enabledLayerCount = 0,
    .ppEnabledLayerNames = nullptr,
    .pEnabledFeatures = nullptr
  };
  VkResult res = vkCreateDevice(state.phys_device, &device_info,
      nullptr, &state.device);
  assert(res == VK_SUCCESS);

  // retrieve our queue
  vkGetDeviceQueue(state.device, state.target_family_index, 0, &state.queue);
}

void prepare_surface_creation(AppState& state) {
  // query surface properties
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(state.phys_device,
      state.surface, &state.surface_caps);
  uint32_t format_count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(state.phys_device, state.surface,
      &format_count, nullptr);
  vector<VkSurfaceFormatKHR> surface_formats(format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(state.phys_device, state.surface, &format_count,
      surface_formats.data());
  uint32_t present_mode_count = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(state.phys_device, state.surface,
      &present_mode_count, nullptr);
  vector<VkPresentModeKHR> present_modes(present_mode_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(state.phys_device, state.surface,
      &present_mode_count, present_modes.data());
  assert(format_count > 0 && present_mode_count > 0);
  printf("surface formats: %d, surface present modes: %d\n",
      format_count, present_mode_count);

  bool found_format = false;
  for (int i = 0; i < format_count; ++i) {
    auto& format = surface_formats[i];
    if (format.format == VK_FORMAT_UNDEFINED ||
        (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
         format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) {
      state.target_format = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
      found_format = true;
      break;
    }
  }
  assert(found_format);

  // guaranteed to be supported?
  state.target_present_mode = VK_PRESENT_MODE_FIFO_KHR;

  state.target_extent = state.surface_caps.currentExtent;
  state.target_image_count = state.surface_caps.minImageCount + 1;
  printf("target extent w: %d, h %d\n", state.target_extent.width,
      state.target_extent.height);
  printf("image count min: %d, max: %d\n",
      state.surface_caps.minImageCount, state.surface_caps.maxImageCount);
}

VkImageView create_image_view(AppState& state, VkImage image,
    VkFormat format, VkImageAspectFlags aspect_flags) {
  VkImageViewCreateInfo view_info = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    .image = image,
    .viewType = VK_IMAGE_VIEW_TYPE_2D,
    .format = format,
    .subresourceRange = {
      .aspectMask = aspect_flags,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1
    }
  };
  VkImageView img_view;
  VkResult res = vkCreateImageView(state.device, &view_info,
      nullptr, &img_view);
  assert(res == VK_SUCCESS);
  return img_view;
}

void setup_swapchain(AppState& state) {
  VkSwapchainCreateInfoKHR swapchain_info = {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    .surface = state.surface,
    .minImageCount = state.target_image_count,
    .imageFormat = state.target_format.format,
    .imageColorSpace = state.target_format.colorSpace,
    .imageExtent = state.target_extent,
    .imageArrayLayers = 1,
    .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = 0,
    .pQueueFamilyIndices = nullptr,
    .preTransform = state.surface_caps.currentTransform,
    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .presentMode = state.target_present_mode,
    .oldSwapchain = VK_NULL_HANDLE
  };
  VkResult res = vkCreateSwapchainKHR(state.device,
      &swapchain_info, nullptr, &state.swapchain);
  assert(res == VK_SUCCESS);

  // retrieve the swapchain images
  uint32_t swapchain_img_count = 0;
  vkGetSwapchainImagesKHR(state.device, state.swapchain,
      &swapchain_img_count, nullptr);
  state.swapchain_images.resize(swapchain_img_count);
  vkGetSwapchainImagesKHR(state.device, state.swapchain,
      &swapchain_img_count, state.swapchain_images.data());

  // create image views
  state.swapchain_img_views.resize(state.swapchain_images.size());
  for (int i = 0; i < state.swapchain_images.size(); ++i) {
    state.swapchain_img_views[i] = create_image_view(state,
        state.swapchain_images[i], state.target_format.format,
        VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

void setup_renderpass(AppState& state) {
  VkSubpassDependency dependency = {
    .srcSubpass = VK_SUBPASS_EXTERNAL,
    .dstSubpass = 0,
    .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    .srcAccessMask = 0,
    .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
  };
  VkAttachmentDescription color_attachment = {
    .format = state.target_format.format,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
  };
  VkAttachmentReference color_attachment_ref = {
    .attachment = 0,
    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  };
  VkAttachmentDescription depth_attachment = {
    .format = find_depth_format(state.phys_device),
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
  };
  VkAttachmentReference depth_attachment_ref = {
    .attachment = 1,
    .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
  };
  VkSubpassDescription subpass_desc = {
    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .colorAttachmentCount = 1,
    .pColorAttachments = &color_attachment_ref,
    .pDepthStencilAttachment = &depth_attachment_ref
  };
  vector<VkAttachmentDescription> attachments = {
    color_attachment, depth_attachment
  };
  VkRenderPassCreateInfo render_pass_info = {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .attachmentCount = (uint32_t) attachments.size(),
    .pAttachments = attachments.data(),
    .subpassCount = 1,
    .pSubpasses = &subpass_desc,
    .dependencyCount = 1,
    .pDependencies = &dependency
  };
  VkResult res = vkCreateRenderPass(state.device, &render_pass_info,
      nullptr, &state.render_pass);
  assert(res == VK_SUCCESS);
}

void setup_descriptor_set_layout(AppState& state) {
  VkDescriptorSetLayoutBinding ubo_layout_binding = {
    .binding = 0,
    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    .pImmutableSamplers = nullptr
  };
  VkDescriptorSetLayoutBinding sampler_layout_binding = {
    .binding = 1,
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    .pImmutableSamplers = nullptr,
    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
  };
  vector<VkDescriptorSetLayoutBinding> bindings = {
    ubo_layout_binding, sampler_layout_binding
  };
  VkDescriptorSetLayoutCreateInfo layout_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = (uint32_t) bindings.size(),
    .pBindings = bindings.data()
  };
  VkResult res = vkCreateDescriptorSetLayout(state.device, &layout_info, nullptr,
      &state.desc_set_layout);
  assert(res == VK_SUCCESS);
}

void setup_graphics_pipeline(AppState& state) {
  auto vert_shader_code = read_file("../shaders/vert.spv");
  auto frag_shader_code = read_file("../shaders/frag.spv");
  VkShaderModule vert_module = create_shader_module(state.device, vert_shader_code);
  VkShaderModule frag_module = create_shader_module(state.device, frag_shader_code);

  VkPipelineShaderStageCreateInfo vert_stage_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_VERTEX_BIT,
    .module = vert_module,
    .pName = "main"
  };
  VkPipelineShaderStageCreateInfo frag_stage_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
    .module = frag_module,
    .pName = "main"
  };
  vector<VkPipelineShaderStageCreateInfo> shader_stages = {
    vert_stage_info, frag_stage_info
  };
  VkPipelineVertexInputStateCreateInfo vertex_input_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &state.binding_desc,
    .vertexAttributeDescriptionCount = (uint32_t) state.attr_descs.size(),
    .pVertexAttributeDescriptions = state.attr_descs.data()
  };
  VkPipelineInputAssemblyStateCreateInfo input_assembly_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    .primitiveRestartEnable = VK_FALSE
  };
  VkViewport viewport = {
    .x = 0.0f,
    .y = 0.0f,
    .width = (float) state.target_extent.width,
    .height = (float) state.target_extent.height,
    .minDepth = 0.0f,
    .maxDepth = 1.0f
  };
  VkRect2D scissor_rect = {
    .offset = {0, 0},
    .extent = state.target_extent
  };
  VkPipelineViewportStateCreateInfo viewport_state_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    .viewportCount = 1,
    .pViewports = &viewport,
    .scissorCount = 1,
    .pScissors = &scissor_rect
  };
  VkPipelineRasterizationStateCreateInfo rast_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .depthClampEnable = VK_FALSE,
    .rasterizerDiscardEnable = VK_FALSE,
    .polygonMode = VK_POLYGON_MODE_FILL,
    .lineWidth = 1.0f,
    .cullMode = VK_CULL_MODE_BACK_BIT,
    .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
    .depthBiasEnable = VK_FALSE,
    .depthBiasConstantFactor = 0.0f,
    .depthBiasClamp = 0.0f,
    .depthBiasSlopeFactor = 0.0f
  };
  VkPipelineMultisampleStateCreateInfo multisampling = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    .sampleShadingEnable = VK_FALSE,
    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    .minSampleShading = 1.0f,
    .pSampleMask = nullptr,
    .alphaToCoverageEnable = VK_FALSE,
    .alphaToOneEnable = VK_FALSE
  };
  VkPipelineColorBlendAttachmentState color_blend_attachment = {
    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    .blendEnable = VK_FALSE,
  };
  VkPipelineColorBlendStateCreateInfo color_blending = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .logicOpEnable = VK_FALSE,
    .attachmentCount = 1,
    .pAttachments = &color_blend_attachment
  };
  VkPipelineDepthStencilStateCreateInfo depth_stencil = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    .depthTestEnable = VK_TRUE,
    .depthWriteEnable = VK_TRUE,
    .depthCompareOp = VK_COMPARE_OP_LESS,
    .depthBoundsTestEnable = VK_FALSE,
    .stencilTestEnable = VK_FALSE
  };
  // descriptor sets for uniforms go here
  VkPipelineLayoutCreateInfo pipeline_layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &state.desc_set_layout
  };
  VkResult res = vkCreatePipelineLayout(state.device, &pipeline_layout_info, nullptr,
      &state.pipeline_layout);
  assert(res == VK_SUCCESS);

  VkGraphicsPipelineCreateInfo graphics_pipeline_info = {
    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    .stageCount = 2,
    .pStages = shader_stages.data(),
    .pVertexInputState = &vertex_input_info,
    .pInputAssemblyState = &input_assembly_info,
    .pViewportState = &viewport_state_info,
    .pRasterizationState = &rast_info,
    .pMultisampleState = &multisampling,
    .pDepthStencilState = &depth_stencil,
    .pColorBlendState = &color_blending,
    .pDynamicState = nullptr,
    .layout = state.pipeline_layout,
    .renderPass = state.render_pass,
    .subpass = 0,
    .basePipelineHandle = VK_NULL_HANDLE,
    .basePipelineIndex = -1
  };
  res = vkCreateGraphicsPipelines(state.device, VK_NULL_HANDLE, 1,
      &graphics_pipeline_info, nullptr, &state.graphics_pipeline);
  assert(res == VK_SUCCESS);

  vkDestroyShaderModule(state.device, vert_module, nullptr);
  vkDestroyShaderModule(state.device, frag_module, nullptr);
}

void setup_framebuffers(AppState& state) {
  state.swapchain_framebuffers.resize(state.swapchain_img_views.size());
  for (int i = 0; i < state.swapchain_img_views.size(); ++i) {
    vector<VkImageView> attachments = {
      state.swapchain_img_views[i], state.depth_img_view
    };
    VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = state.render_pass,
      .attachmentCount = (uint32_t) attachments.size(),
      .pAttachments = attachments.data(),
      .width = state.target_extent.width,
      .height = state.target_extent.height,
      .layers = 1
    };
    VkResult res = vkCreateFramebuffer(state.device, &framebuffer_info, nullptr,
        &state.swapchain_framebuffers[i]);
    assert(res == VK_SUCCESS);
  }
}

void setup_command_pool(AppState& state) {
  VkCommandPoolCreateInfo cmd_pool_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .pNext = nullptr,
    .queueFamilyIndex = state.target_family_index,
    .flags = 0
  };
  VkResult res = vkCreateCommandPool(state.device, &cmd_pool_info, nullptr,
      &state.cmd_pool);
  assert(res == VK_SUCCESS);
}

void create_image(AppState& state, uint32_t w, uint32_t h,
    VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags mem_props, VkImage& image,
    VkDeviceMemory& image_mem) {

  VkImageCreateInfo img_info = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .imageType = VK_IMAGE_TYPE_2D,
    .extent.width = w,
    .extent.height = h,
    .extent.depth = 1,
    .mipLevels = 1,
    .arrayLayers = 1,
    .format = format,
    .tiling = tiling,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .flags = 0
  };
  VkResult res = vkCreateImage(state.device, &img_info,
      nullptr, &image);
  assert(res == VK_SUCCESS);

  VkMemoryRequirements mem_reqs;
  vkGetImageMemoryRequirements(state.device, image, &mem_reqs);
  VkMemoryAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = mem_reqs.size,
    .memoryTypeIndex = find_mem_type_index(state.phys_device,
        mem_reqs.memoryTypeBits, mem_props)
  };
  res = vkAllocateMemory(state.device, &alloc_info, nullptr,
    &image_mem);
  assert(res == VK_SUCCESS);

  vkBindImageMemory(state.device, image, image_mem, 0); 
}

void copy_buffer_to_image(AppState& state, VkBuffer buffer,
    VkImage image, uint32_t w, uint32_t h) {
  VkCommandBuffer tmp_cmd_buffer = begin_single_time_commands(state);

  VkBufferImageCopy region = {
    .bufferOffset = 0,
    .bufferRowLength = 0,
    .bufferImageHeight = 0,
    .imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .imageSubresource.mipLevel = 0,
    .imageSubresource.baseArrayLayer = 0,
    .imageSubresource.layerCount = 1,
    .imageOffset = {0, 0, 0},
    .imageExtent = {w, h, 1}
  };
  vkCmdCopyBufferToImage(tmp_cmd_buffer, buffer, image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  end_single_time_commands(state, tmp_cmd_buffer);
}

void transition_image_layout(AppState& state, VkImage img,
    VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout) {
  VkCommandBuffer tmp_cmd_buffer = begin_single_time_commands(state);

  VkAccessFlags src_access, dst_access;
  VkPipelineStageFlags src_stage, dst_stage;

  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    src_access = 0;
    dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
      new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    dst_access = VK_ACCESS_SHADER_READ_BIT;
    src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    src_access = 0;
    dst_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition");
  }

  VkImageMemoryBarrier barrier = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    .oldLayout = old_layout,
    .newLayout = new_layout,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .image = img,
    .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .subresourceRange.baseMipLevel = 0,
    .subresourceRange.levelCount = 1,
    .subresourceRange.baseArrayLayer = 0,
    .subresourceRange.layerCount = 1,
    .srcAccessMask = src_access,
    .dstAccessMask = dst_access
  };
  if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (has_stencil_component(format)) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  }
  vkCmdPipelineBarrier(tmp_cmd_buffer,
      src_stage, dst_stage,
      0,
      0, nullptr,
      0, nullptr,
      1, &barrier);

  end_single_time_commands(state, tmp_cmd_buffer);
}

void setup_texture_image(AppState& state) {
  int tex_w, tex_h, tex_channels;
  stbi_uc* pixels = stbi_load("../textures/sample_tex.jpg",
      &tex_w, &tex_h, &tex_channels, STBI_rgb_alpha);
  VkDeviceSize img_size = tex_w * tex_h * 4;
  assert(pixels);

  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_mem;

  create_buffer(state.device, state.phys_device, img_size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      staging_buffer, staging_buffer_mem);
  void* data;
  vkMapMemory(state.device, staging_buffer_mem, 0, img_size, 0, &data);
  memcpy(data, pixels, (size_t) img_size);
  vkUnmapMemory(state.device, staging_buffer_mem);
  
  stbi_image_free(pixels);

  create_image(state, tex_w, tex_h, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      state.texture_img, state.texture_img_mem);

  transition_image_layout(state, state.texture_img,
      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  copy_buffer_to_image(state, staging_buffer, state.texture_img, (uint32_t) tex_w,
      (uint32_t) tex_h);
  transition_image_layout(state, state.texture_img,
      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  vkDestroyBuffer(state.device, staging_buffer, nullptr);
  vkFreeMemory(state.device, staging_buffer_mem, nullptr);
}

void setup_texture_image_view(AppState& state) {
  state.texture_img_view = create_image_view(state,
      state.texture_img, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_ASPECT_COLOR_BIT);
}

void setup_texture_sampler(AppState& state) {
  VkSamplerCreateInfo sampler_info = {
    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    .magFilter = VK_FILTER_LINEAR,
    .minFilter = VK_FILTER_LINEAR,
    .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    .anisotropyEnable = VK_FALSE,
    .maxAnisotropy = 1,
    .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    .unnormalizedCoordinates = VK_FALSE,
    .compareEnable = VK_FALSE,
    .compareOp = VK_COMPARE_OP_ALWAYS,
    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
    .mipLodBias = 0.0f,
    .minLod = 0.0f,
    .maxLod = 0.0f
  };
  VkResult res = vkCreateSampler(state.device, &sampler_info,
      nullptr, &state.texture_sampler);
  assert(res == VK_SUCCESS);
}

void setup_depth_resources(AppState& state) {
  VkFormat depth_format = find_depth_format(state.phys_device);

  create_image(state, state.target_extent.width, state.target_extent.height,
      depth_format, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, state.depth_img,
      state.depth_img_mem);
  state.depth_img_view = create_image_view(state, state.depth_img,
      depth_format, VK_IMAGE_ASPECT_DEPTH_BIT);

  transition_image_layout(state, state.depth_img,
      depth_format, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void setup_vertex_buffer(AppState& state, vector<Vertex>& vertices) {
  VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_mem;
  create_buffer(state.device, state.phys_device,
      buffer_size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      staging_buffer, staging_buffer_mem);

  create_buffer(state.device, state.phys_device,
      buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      state.vert_buffer, state.vert_buffer_mem);

  // upload vertex data to vertex buffer mem
  void* mapped_data;
  vkMapMemory(state.device, staging_buffer_mem, 0, buffer_size, 0, 
    &mapped_data);
  memcpy(mapped_data, vertices.data(), (size_t) buffer_size);
  vkUnmapMemory(state.device, staging_buffer_mem);

  copy_buffer(state, staging_buffer,
      state.vert_buffer, buffer_size);

  vkDestroyBuffer(state.device, staging_buffer, nullptr);
  vkFreeMemory(state.device, staging_buffer_mem, nullptr);
}

void setup_index_buffer(AppState& state, vector<uint16_t>& indices) {
  create_index_buffer(state,
      indices, state.index_buffer, state.index_buffer_mem);
}

void setup_uniform_buffers(AppState& state) {
  state.unif_buffers.resize(state.swapchain_img_views.size());
  state.unif_buffers_mem.resize(state.swapchain_img_views.size());
  VkDeviceSize unif_buffer_size = sizeof(UniformBufferObject);
  for (size_t i = 0; i < state.unif_buffers.size(); ++i) {
    create_buffer(state.device, state.phys_device, unif_buffer_size,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        state.unif_buffers[i], state.unif_buffers_mem[i]);
  }
}

void setup_descriptor_pool(AppState& state) {
  vector<VkDescriptorPoolSize> pool_sizes(2);
  pool_sizes[0] = {
    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    .descriptorCount = (uint32_t) state.swapchain_img_views.size()
  };
  pool_sizes[1] = {
    .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    .descriptorCount = (uint32_t) state.swapchain_img_views.size()
  };
  VkDescriptorPoolCreateInfo desc_pool_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .poolSizeCount = (uint32_t) pool_sizes.size(),
    .pPoolSizes = pool_sizes.data(),
    .maxSets = (uint32_t) state.swapchain_img_views.size()
  };
  VkResult res = vkCreateDescriptorPool(state.device, &desc_pool_info,
      nullptr, &state.desc_pool);
  assert(res == VK_SUCCESS);
}

void setup_descriptor_sets(AppState& state) {
  vector<VkDescriptorSetLayout> desc_set_layouts(
      state.swapchain_img_views.size(), state.desc_set_layout);
  VkDescriptorSetAllocateInfo desc_set_alloc_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = state.desc_pool,
    .descriptorSetCount = (uint32_t) desc_set_layouts.size(),
    .pSetLayouts = desc_set_layouts.data()
  };
  state.desc_sets.resize(desc_set_layouts.size());
  VkResult res = vkAllocateDescriptorSets(state.device,
      &desc_set_alloc_info, state.desc_sets.data());
  assert(res == VK_SUCCESS);
  
  for (size_t i = 0; i < state.desc_sets.size(); ++i) {
    VkDescriptorBufferInfo buffer_info = {
      .buffer = state.unif_buffers[i],
      .offset = 0,
      .range = sizeof(UniformBufferObject)
    };
    VkDescriptorImageInfo image_info = {
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .imageView = state.texture_img_view,
      .sampler = state.texture_sampler
    };
    vector<VkWriteDescriptorSet> desc_writes(2);
    desc_writes[0] = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = state.desc_sets[i],
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .pBufferInfo = &buffer_info,
      .pImageInfo = nullptr,
      .pTexelBufferView = nullptr
    };
    desc_writes[1] = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = state.desc_sets[i],
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .pImageInfo = &image_info
    };
    vkUpdateDescriptorSets(state.device, (uint32_t) desc_writes.size(),
        desc_writes.data(), 0, nullptr);
  }
}

void setup_command_buffers(AppState& state) {
  state.cmd_buffers.resize(state.swapchain_framebuffers.size());
  VkCommandBufferAllocateInfo cmd_buffer_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = state.cmd_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = (uint32_t) state.swapchain_framebuffers.size()
  };
  VkResult res = vkAllocateCommandBuffers(state.device, &cmd_buffer_info,
      state.cmd_buffers.data());
  assert(res == VK_SUCCESS);
}

void record_render_passes(AppState& state, vector<uint16_t>& indices) {
  for (int i = 0; i < state.cmd_buffers.size(); ++i) {
    VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
      .pInheritanceInfo = nullptr
    };
    VkResult res = vkBeginCommandBuffer(state.cmd_buffers[i], &begin_info);
    assert(res == VK_SUCCESS);

    array<VkClearValue, 2> clear_values = {};
    clear_values[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_pass_info = {
      .sType= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = state.render_pass,
      .framebuffer = state.swapchain_framebuffers[i],
      .renderArea.offset = {0, 0},
      .renderArea.extent = state.target_extent,
      .clearValueCount = (uint32_t) clear_values.size(),
      .pClearValues = clear_values.data()
    };
    vector<VkBuffer> vert_buffers = {state.vert_buffer};
    vector<VkDeviceSize> byte_offsets = {0};

    vkCmdBeginRenderPass(state.cmd_buffers[i], &render_pass_info,
        VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(state.cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
        state.graphics_pipeline);
    vkCmdBindVertexBuffers(state.cmd_buffers[i], 0, vert_buffers.size(),
        vert_buffers.data(), byte_offsets.data());
    vkCmdBindIndexBuffer(state.cmd_buffers[i], state.index_buffer, 0,
        VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(state.cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
        state.pipeline_layout, 0, 1, &state.desc_sets[i], 0, nullptr);
    //vkCmdDraw(cmd_buffers[i], (uint32_t) vertices.size(), 1, 0, 0);
    vkCmdDrawIndexed(state.cmd_buffers[i], (uint32_t) indices.size(),
        1, 0, 0, 0);
    vkCmdEndRenderPass(state.cmd_buffers[i]);
    res = vkEndCommandBuffer(state.cmd_buffers[i]);
    assert(res == VK_SUCCESS);
  }
}

void setup_sync_objects(AppState& state) {
  state.img_available_semas.resize(max_frames_in_flight);
  state.render_done_semas.resize(max_frames_in_flight);
  state.in_flight_fences.resize(max_frames_in_flight);
  VkSemaphoreCreateInfo sema_info = {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
  };
  VkFenceCreateInfo fence_info = {
    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    .flags = VK_FENCE_CREATE_SIGNALED_BIT
  };
  for (int i = 0; i < max_frames_in_flight; ++i) {
    VkResult res = vkCreateSemaphore(state.device, &sema_info, nullptr,
        &state.img_available_semas[i]);
    assert(res == VK_SUCCESS);
    res = vkCreateSemaphore(state.device, &sema_info, nullptr,
        &state.render_done_semas[i]);
    assert(res == VK_SUCCESS);
    res = vkCreateFence(state.device, &fence_info, nullptr,
        &state.in_flight_fences[i]);
    assert(res == VK_SUCCESS);
  }
}

void render_frame(AppState& state) {
  size_t current_frame = state.current_frame;

  vkWaitForFences(state.device, 1, &state.in_flight_fences[current_frame],
        VK_TRUE, std::numeric_limits<uint64_t>::max());
  vkResetFences(state.device, 1, &state.in_flight_fences[current_frame]);
  
  uint32_t img_index;
  vkAcquireNextImageKHR(state.device, state.swapchain,
      std::numeric_limits<uint64_t>::max(),
      state.img_available_semas[current_frame],
      VK_NULL_HANDLE, &img_index);
  
  // update the unif buffers
  mat4 model_mat = mat4(1.0f);
  mat4 view_mat = glm::lookAt(vec3(2.0f), vec3(0.0f),
      vec3(0.0f, 1.0f, 0.0f));
  float aspect_ratio = state.target_extent.width / (float) state.target_extent.height;
  mat4 proj_mat = glm::perspective(45.0f, aspect_ratio, 0.1f, 10.0f);
  // invert Y b/c vulkan's y-axis is inverted wrt OpenGL
	proj_mat[1][1] *= -1;
  UniformBufferObject ubo = {
    .model = model_mat,
    .view = view_mat,
    .proj = proj_mat
  };
  void* unif_data;
  vkMapMemory(state.device, state.unif_buffers_mem[img_index], 0,
      sizeof(ubo), 0, &unif_data);
  memcpy(unif_data, &ubo, sizeof(ubo));
  vkUnmapMemory(state.device, state.unif_buffers_mem[img_index]);

  // submit cmd buffer to pipeline
  vector<VkPipelineStageFlags> wait_stages = {
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
  };
  VkSubmitInfo submit_info = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &state.img_available_semas[current_frame],
    .pWaitDstStageMask = wait_stages.data(),
    .commandBufferCount = 1,
    .pCommandBuffers = &state.cmd_buffers[img_index],
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &state.render_done_semas[current_frame]
  };
  VkResult res = vkQueueSubmit(state.queue, 1, &submit_info,
      state.in_flight_fences[current_frame]);
  assert(res == VK_SUCCESS);

  // present result when done
  VkPresentInfoKHR present_info = {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &state.render_done_semas[current_frame],
    .swapchainCount = 1,
    .pSwapchains = &state.swapchain,
    .pImageIndices = &img_index,
    .pResults = nullptr
  };
  vkQueuePresentKHR(state.queue, &present_info);

  state.current_frame = (current_frame + 1) % max_frames_in_flight;
}

void cleanup_vulkan(AppState& state) {
  for (size_t i = 0; i < state.swapchain_img_views.size(); ++i) {
    vkDestroyBuffer(state.device, state.unif_buffers[i], nullptr);
    vkFreeMemory(state.device, state.unif_buffers_mem[i], nullptr);
  }
  vkDestroyDescriptorPool(state.device, state.desc_pool, nullptr);
  vkDestroyDescriptorSetLayout(state.device, state.desc_set_layout, nullptr);
  vkDestroyBuffer(state.device, state.index_buffer, nullptr);
  vkFreeMemory(state.device, state.index_buffer_mem, nullptr);
  vkDestroyBuffer(state.device, state.vert_buffer, nullptr);
  vkFreeMemory(state.device, state.vert_buffer_mem, nullptr);
  vkDestroySampler(state.device, state.texture_sampler, nullptr);
  vkDestroyImageView(state.device, state.texture_img_view, nullptr);
  vkDestroyImage(state.device, state.texture_img, nullptr);
  vkFreeMemory(state.device, state.texture_img_mem, nullptr);
  for (int i = 0; i < max_frames_in_flight; ++i) {
    vkDestroySemaphore(state.device, state.render_done_semas[i], nullptr);
    vkDestroySemaphore(state.device, state.img_available_semas[i], nullptr);
    vkDestroyFence(state.device, state.in_flight_fences[i], nullptr);
  }
  vkDestroyCommandPool(state.device, state.cmd_pool, nullptr);
  for (VkFramebuffer& fb : state.swapchain_framebuffers) {
    vkDestroyFramebuffer(state.device, fb, nullptr);
  }
  vkDestroyImageView(state.device, state.depth_img_view, nullptr);
  vkDestroyImage(state.device, state.depth_img, nullptr);
  vkFreeMemory(state.device, state.depth_img_mem, nullptr);
  for (VkImageView& img_view : state.swapchain_img_views) {
    vkDestroyImageView(state.device, img_view, nullptr);
  }
  vkDestroyRenderPass(state.device, state.render_pass, nullptr);
  vkDestroyPipeline(state.device, state.graphics_pipeline, nullptr);
  vkDestroyPipelineLayout(state.device, state.pipeline_layout, nullptr);
  vkDestroySwapchainKHR(state.device, state.swapchain, nullptr);
  state.destroy_debug_utils(state.inst, state.debug_messenger, nullptr);
  vkDestroyDevice(state.device, nullptr);
  vkDestroySurfaceKHR(state.inst, state.surface, nullptr);
  vkDestroyInstance(state.inst, nullptr);
}

void init_vulkan(AppState& state) {
  VkResult res;

  vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

		{{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
  };
  vector<uint16_t> indices = {
    0, 1, 2,
    2, 3, 0,
		4, 5, 6,
		6, 7, 4
  };
  
  setup_vertex_attr_desc(state);
  setup_instance(state);
  setup_debug_callback(state);
  setup_surface(state); 
  setup_physical_device(state);
  setup_logical_device(state);
  prepare_surface_creation(state);
  setup_swapchain(state);
  setup_renderpass(state);
  setup_descriptor_set_layout(state);
  setup_graphics_pipeline(state);
  setup_command_pool(state);
  setup_texture_image(state);
  setup_texture_image_view(state);
  setup_texture_sampler(state);
  setup_depth_resources(state);
  setup_framebuffers(state);
  setup_vertex_buffer(state, vertices);
  setup_index_buffer(state, indices);
  setup_uniform_buffers(state);
  setup_descriptor_pool(state);
  setup_descriptor_sets(state);
  setup_command_buffers(state);
  record_render_passes(state, indices);
  setup_sync_objects(state);
}

void init_glfw(AppState& state) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  // prevent resizing the window for now, since
  // it requires recreating the swapchain
  // TODO: https://vulkan-tutorial.com/Drawing_a_triangle/Swap_chain_recreation
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  state.win = glfwCreateWindow(800, 600, "morph",
      nullptr, nullptr);
  assert(glfwVulkanSupported() == GLFW_TRUE);
}

void main_loop(AppState& state) {
  state.current_frame = 0;
  while (!glfwWindowShouldClose(state.win)) {
    glfwPollEvents();

    render_frame(state);
  }
  vkDeviceWaitIdle(state.device);
}

void cleanup_state(AppState& state) {
  cleanup_vulkan(state);

  glfwDestroyWindow(state.win);
  glfwTerminate();
}

void run_app(int argc, char** argv) {
  // setup the environment for the vulkan loader
  char icd_env_entry[] = ENV_VK_ICD_FILENAMES;
  char layer_env_entry[] = ENV_VK_LAYER_PATH;
  putenv(icd_env_entry);
  putenv(layer_env_entry);

  AppState state;

  init_glfw(state);
  init_vulkan(state);
  main_loop(state);
  cleanup_state(state);
}

/*
void run_app(int argc, char** argv) {
  signal(SIGSEGV, handle_segfault);

  // read cmd-line args
  if (argc < 2) {
    printf("Incorrect usage. Please use:\n\nexec path\n\n"
        "where \"path\" is the path to the directory containing the"
        " \"shaders\" folder. Ex: \"exec ..\"\n");
    return;
  }
  string base_shader_path(argv[1]);

  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    exit(EXIT_FAILURE);
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  int target_width = 1300;
  int target_height = 700;
  GLFWwindow* window = glfwCreateWindow(target_width, target_height, "morph", NULL, NULL);
  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window);
  gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
  glfwSwapInterval(1);

  printf("OpenGL: %d.%d\n", GLVersion.major, GLVersion.minor);

  GraphicsState g_state(window, base_shader_path);
  setup_opengl(g_state);

  glfwSetWindowUserPointer(window, &g_state);
  glfwSetKeyCallback(window, handle_key_event);

  // setup imgui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  ImGui::StyleColorsDark();
  //ImGui::StyleColorsClassic();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  const char* glsl_version = "#version 150";
  ImGui_ImplOpenGL3_Init(glsl_version);

  bool requires_mac_mojave_fix = true;
  bool show_dev_console = true;
  // for maintaining the fps
  auto start_of_frame = chrono::steady_clock::now();
  // for logging the fps
  int fps_stat = 0;
  int frame_counter = 0;
  auto start_of_sec = start_of_frame;

  while (!glfwWindowShouldClose(window)) {
    std::this_thread::sleep_until(start_of_frame);
    auto cur_time = chrono::steady_clock::now();
    int frame_dur_millis = (int) (1000.0f / g_state.controls.target_fps);
    start_of_frame = cur_time + chrono::milliseconds(frame_dur_millis);

    // for logging the fps
    frame_counter += 1;
    if (start_of_sec < cur_time) {
      start_of_sec = cur_time + chrono::seconds(1);
      fps_stat = frame_counter;
      frame_counter = 0;
    }

    glfwPollEvents();
    update_camera(window, g_state.controls, g_state.camera);

    // the screen appears black on mojave until a resize occurs
    if (requires_mac_mojave_fix) {
      requires_mac_mojave_fix = false;
      glfwSetWindowSize(window, target_width, target_height + 1);
    }
    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("dev console", &show_dev_console);
    ImGui::DragInt("target fps", &g_state.controls.target_fps, 1.0f, 0, 100);
    ImGui::Text("current fps: %d", fps_stat);

    Controls& controls = g_state.controls;

    ImGui::Separator();
    ImGui::Text("camera:");
    ImGui::Text("eye: %s", vec3_str(g_state.camera.pos()).c_str());
    ImGui::Text("forward: %s", vec3_str(g_state.camera.forward()).c_str());
    ImGui::Text("mode: %s", controls.cam_spherical_mode ? "spherical" : "cartesian");

    ImGui::Separator();
    ImGui::Text("render controls:");
    ImGui::Checkbox("render faces", &controls.render_faces);
    ImGui::Checkbox("render points", &controls.render_points);
    ImGui::Checkbox("render wireframe", &controls.render_wireframe);

    ImGui::Separator();
    ImGui::Text("render program controls:");
    ImGui::PushID("render");
    gen_user_uniforms_ui(g_state.render_state.prog.user_unifs);
    ImGui::PopID();

    ImGui::Separator();
    ImGui::Text("morph program controls:");
    MorphState& m_state = g_state.morph_state;

    vector<const char*> prog_names;
    for (auto& prog : g_state.morph_state.programs) {
      prog_names.push_back(prog.name.c_str());
    }
    ImGui::Combo("program", &m_state.cur_prog_index,
        prog_names.data(), prog_names.size());

    MorphProgram& cur_prog = m_state.programs[m_state.cur_prog_index];
    ImGui::PushID("morph");
    gen_user_uniforms_ui(cur_prog.user_unifs);
    ImGui::PopID();
    
    ImGui::Separator();
    ImGui::Text("simulation controls:");
    ImGui::Text("init data:");
    ImGui::InputInt("AxA samples", &controls.num_zygote_samples);
    controls.num_zygote_samples = std::max(controls.num_zygote_samples, 0);
    
    ImGui::Text("simulation:"); 
    int max_iter_num = 1*1000*1000*1000;
    ImGui::DragInt("iter num", &controls.num_iters, 0.2f, 0, max_iter_num);
    if (ImGui::Button("run once")) {
      run_simulation_pipeline(g_state);  
    }
    ImGui::Text("animation:");
    string anim_btn_text(controls.animating_sim ? "PAUSE" : "PLAY");
    if (ImGui::Button(anim_btn_text.c_str())) {
      controls.animating_sim = !controls.animating_sim;
    }
    ImGui::DragInt("start iter", &controls.start_iter_num, 10.0f, 0, max_iter_num);
    ImGui::DragInt("end iter", &controls.end_iter_num, 10.0f, controls.start_iter_num, max_iter_num);
    ImGui::DragInt("delta iters per frame", &controls.delta_iters, 0.2f, -10, 10);
    ImGui::Checkbox("loop at end", &controls.loop_at_end);
    
    // run the animation
    if (controls.animating_sim) {
      controls.num_iters += controls.delta_iters;
      controls.num_iters = clamp(controls.num_iters,
          controls.start_iter_num, controls.end_iter_num);
      if (controls.loop_at_end && controls.num_iters == controls.end_iter_num) {
        controls.num_iters = controls.start_iter_num;
      }
      run_simulation_pipeline(g_state);
    }

    ImGui::Separator();
    ImGui::Text("debug");
    ImGui::Text("Note that logging will not occur while animating");
    ImGui::Checkbox("log input nodes", &controls.log_input_nodes);
    ImGui::Checkbox("log output nodes", &controls.log_output_nodes);
    ImGui::Checkbox("log render data", &controls.log_render_data);
    ImGui::Checkbox("log durations", &controls.log_durations);
    // do not log while animating, the IO becomes a bottleneck
    if (controls.animating_sim) {
      controls.log_input_nodes = false;
      controls.log_output_nodes = false;
      controls.log_render_data = false;
      controls.log_durations = false;
    }

    ImGui::Separator();
    ImGui::Text("instructions:");
    ImGui::Text("%s", INSTRUCTIONS_STRING);

    ImGui::End();

    ImGui::Render();
    int fb_width = 0;
    int fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    
    glViewport(0, 0, fb_width, fb_height);
    glClearColor(1, 1, 1, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    g_state.render_state.fb_width = fb_width;
    g_state.render_state.fb_height = fb_height;
    render_frame(g_state);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}*/
