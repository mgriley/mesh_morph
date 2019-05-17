#include "app.h"
#include "utils.h"
#include "types.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <shaderc/shaderc.hpp>

#include <chrono>
#include <thread>
#include <utility>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

const uint32_t MAX_NUM_VERTICES = (int) 1e6;
const uint32_t MAX_NUM_INDICES = (int) 1e6;

const uint32_t LOCAL_WORKGROUP_SIZE = 256;

const int max_frames_in_flight = 2;

static void check_vk_result(VkResult res) {
  assert(res == VK_SUCCESS);
}

vector<char> read_file(const string& filename) {
  ifstream file(filename, ios::ate | ios::binary);
  if (!file.is_open()) {
    printf("Could not open: %s\n", filename.c_str());
    throw std::runtime_error("file not found");
  }
  size_t file_size = (size_t) file.tellg();
  vector<char> buffer(file_size);
  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();
  return buffer;
}

vector<UserUnif> parse_user_unifs(const string& shader_text) {
  vector<UserUnif> unifs;

  string block_start = "// BEGIN_USER_UNIFS";
  string block_end = "// END_USER_UNIFS";

  const char* s = shader_text.c_str();
  bool in_block = false;
  while (true) {
    if (!in_block) {
      // until the start block is reached, progress a line at a time
      if (strncmp(s, block_start.c_str(), block_start.size()) == 0) {
        in_block = true;
      }
      s = strchr(s, '\n');
      if (!s) {
        break;
      }
      s += 1;
    } else {
      // until the end block is reached, read one comment line of specification followed
      // by one line with the vec4 itself
      if (strncmp(s, block_end.c_str(), block_end.size()) == 0) {
        break;
      }
      int num_comps = 4;
      float min_val = 0.0;
      float max_val = 1.0;
      float drag_speed = 1.0;
      vec4 def_val = vec4(0.0);
      int num_matches = sscanf(s,
          " // comps %d min %f max %f speed %f def %f %f %f %f",
          &num_comps, &min_val, &max_val, &drag_speed,
          &def_val[0], &def_val[1], &def_val[2], &def_val[3]);
      assert(num_matches >= 1);
      s = strchr(s, '\n') + 1;

      // Note: name will actually include the ; char in this case
      char name[100];
      num_matches = sscanf(s,
        " vec4 %s;", name);
      assert(num_matches == 1);
      s = strchr(s, '\n') + 1;

      UserUnif unif(name, num_comps, def_val,
          min_val, max_val, drag_speed);
      unifs.push_back(unif);
    } 
  }
  return unifs;
}

/*
   Return spirv text and parses any user uniforms
*/
vector<uint32_t> process_shader_file(
    const string& src_name,
    const string& filename,
    shaderc_shader_kind kind,
    vector<UserUnif>& out_unifs) {
  using namespace shaderc;
  Compiler compiler;
  CompileOptions compile_options;

  vector<char> glsl_source_vec = read_file(filename);
  string glsl_source(glsl_source_vec.begin(), glsl_source_vec.end());

  SpvCompilationResult res = compiler.CompileGlslToSpv(
      glsl_source, kind, src_name.c_str(), compile_options);
  if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
    printf("Compilation error:\n%s\n", res.GetErrorMessage().c_str());
    return vector<uint32_t>();
  }
  out_unifs = parse_user_unifs(glsl_source);

  return vector<uint32_t>(res.cbegin(), res.cend());
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

VkShaderModule create_shader_module(VkDevice& device,
    const vector<uint32_t>& code) {
  VkShaderModuleCreateInfo create_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = sizeof(code[0]) * code.size(),
    .pCode = code.data()
  };
  VkShaderModule module;
  VkResult res = vkCreateShaderModule(device, &create_info,
      nullptr, &module);
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

// TODO - move all the vulkan helper stuff to a diff file
struct StagingBuf {
  VkBuffer buf;
  VkDeviceMemory mem;

  StagingBuf(AppState& state, uint32_t buffer_size);
  void cleanup(AppState& state);
};

StagingBuf::StagingBuf(AppState& state, uint32_t buffer_size)
{
  create_buffer(state.device, state.phys_device, buffer_size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      buf, mem);
}

void StagingBuf::cleanup(AppState& state) {
  vkDestroyBuffer(state.device, buf, nullptr);
  vkFreeMemory(state.device, mem, nullptr);
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
  state.vert_binding_descs.clear();
  for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
    VkVertexInputBindingDescription binding_desc = {
      .binding = i,
      .stride = sizeof(vec4),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
    };
    state.vert_binding_descs.push_back(binding_desc);
  }
  state.vert_attr_descs.clear();
  for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
    VkVertexInputAttributeDescription attr_desc = {
      .binding = i,
      .location = i,
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = 0
    };
    state.vert_attr_descs.push_back(attr_desc);
  }
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

void prepare_swapchain_creation(AppState& state) {
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
  prepare_swapchain_creation(state);

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

void setup_render_desc_set_layout(AppState& state) {
  vector<VkDescriptorSetLayoutBinding> layout_bindings;
  for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
    VkDescriptorSetLayoutBinding binding = {
      .binding = i,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .pImmutableSamplers = nullptr
    };
    layout_bindings.push_back(binding);
  }
  VkDescriptorSetLayoutCreateInfo layout_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = (uint32_t) layout_bindings.size(),
    .pBindings = layout_bindings.data()
  };
  VkResult res = vkCreateDescriptorSetLayout(state.device,
      &layout_info, nullptr, &state.render_desc_set_layout);
  assert(res == VK_SUCCESS);
}

void setup_compute_desc_set_layout(AppState& state) {
  // TODO - add in the bindings for the shared shader storage

  vector<VkDescriptorSetLayoutBinding> bindings;
  // we need a texel buffer for each input attr and a buffer
  // for each output attribute
  for (uint32_t i = 0; i < 2 * ATTRIBUTES_COUNT; ++i) {
    VkDescriptorSetLayoutBinding binding = {
      .binding = i,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .pImmutableSamplers = nullptr
    };
    bindings.push_back(binding);
  }
  // shared storage buffer
  VkDescriptorSetLayoutBinding storage_binding = {
    .binding = 2 * ATTRIBUTES_COUNT,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .pImmutableSamplers = nullptr
  };
  bindings.push_back(storage_binding);

  VkDescriptorSetLayoutCreateInfo layout_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = (uint32_t) bindings.size(),
    .pBindings = bindings.data()
  };
  VkResult res = vkCreateDescriptorSetLayout(state.device,
      &layout_info, nullptr, &state.compute_desc_set_layout);
  assert(res == VK_SUCCESS);
}

void setup_desc_set_layouts(AppState& state) {
  setup_render_desc_set_layout(state);
  setup_compute_desc_set_layout(state);
}

void setup_graphics_pipelines(AppState& state) {
  vector<UserUnif> vertex_unifs, frag_unifs;
  auto vert_shader_code = process_shader_file(
      "vertex shader", "../shaders/basic.vert",
      shaderc_glsl_vertex_shader, vertex_unifs);
  auto frag_shader_code = process_shader_file(
      "frag shader", "../shaders/basic.frag",
      shaderc_glsl_fragment_shader, frag_unifs);
  VkShaderModule vert_module = create_shader_module(state.device, vert_shader_code);
  VkShaderModule frag_module = create_shader_module(state.device, frag_shader_code);
  // TODO - add on the frag unifs?
  state.render_unifs = std::move(vertex_unifs);

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
    .vertexBindingDescriptionCount = (uint32_t) state.vert_binding_descs.size(),
    .pVertexBindingDescriptions = state.vert_binding_descs.data(),
    .vertexAttributeDescriptionCount = (uint32_t) state.vert_attr_descs.size(),
    .pVertexAttributeDescriptions = state.vert_attr_descs.data()
  };
  array<VkPrimitiveTopology, PIPELINES_COUNT> topologies = {
    VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
  };
  array<VkPipelineInputAssemblyStateCreateInfo, PIPELINES_COUNT> input_assembly_infos;
  for (uint32_t i = 0; i < PIPELINES_COUNT; ++i) {
    VkPipelineInputAssemblyStateCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topologies[i],
      .primitiveRestartEnable = VK_FALSE
    };
    input_assembly_infos[i] = info;
  }
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
  // Only allow push constants in the vertex shader for now
  VkPushConstantRange push_constant_range = {
    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    .offset = 0,
    .size = (uint32_t) sizeof(RenderPushConstants)
  };
  VkPipelineLayoutCreateInfo pipeline_layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &state.render_desc_set_layout,
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_constant_range
  };
  VkResult res = vkCreatePipelineLayout(state.device,
      &pipeline_layout_info, nullptr,
      &state.render_pipeline_layout);
  assert(res == VK_SUCCESS);
  
  // create a pipeline for each type of input assembly
  // designate the first pipeline as the parent and the rest
  // will be derivatives. this may result in optimizations, unsure
  array<VkGraphicsPipelineCreateInfo, PIPELINES_COUNT> graphics_pipeline_infos;
  for (uint32_t i = 0; i < PIPELINES_COUNT; ++i) {
    VkPipelineCreateFlags flags = i == 0 ?
      VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT :
      VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .flags = flags,
      .stageCount = (uint32_t) shader_stages.size(),
      .pStages = shader_stages.data(),
      .pVertexInputState = &vertex_input_info,
      .pInputAssemblyState = &input_assembly_infos[i],
      .pViewportState = &viewport_state_info,
      .pRasterizationState = &rast_info,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &color_blending,
      .pDynamicState = nullptr,
      .layout = state.render_pipeline_layout,
      .renderPass = state.render_pass,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = 0
    };
    graphics_pipeline_infos[i] = info;
  }
  res = vkCreateGraphicsPipelines(state.device, VK_NULL_HANDLE,
      (uint32_t) graphics_pipeline_infos.size(),
      graphics_pipeline_infos.data(), nullptr,
      state.graphics_pipelines.data());
  assert(res == VK_SUCCESS);

  vkDestroyShaderModule(state.device, vert_module, nullptr);
  vkDestroyShaderModule(state.device, frag_module, nullptr);
}

void setup_compute_pipeline(AppState& state) {
  auto shader_code = process_shader_file(
      "compute shader", "../shaders/morph.comp",
      shaderc_glsl_compute_shader, state.compute_unifs);
  VkShaderModule shader_module = create_shader_module(
      state.device, shader_code);

  vector<uint32_t> spec_data = {LOCAL_WORKGROUP_SIZE};
  vector<VkSpecializationMapEntry> spec_entries = {
    {1, 0, sizeof(uint32_t)}
  };
  VkSpecializationInfo spec_info = {
    .mapEntryCount = (uint32_t) spec_entries.size(),
    .pMapEntries = spec_entries.data(),
    .dataSize = sizeof(spec_data[0]) * spec_data.size(),
    .pData = spec_data.data()
  };
  VkPipelineShaderStageCreateInfo stage_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = shader_module,
    .pName = "main",
    .pSpecializationInfo = &spec_info
  };
  VkPushConstantRange push_constant_range = {
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = (uint32_t) sizeof(ComputePushConstants)
  };
  VkPipelineLayoutCreateInfo pipeline_layout_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &state.compute_desc_set_layout,
    .pushConstantRangeCount = 1,
    .pPushConstantRanges = &push_constant_range
  };
  VkResult res = vkCreatePipelineLayout(state.device,
      &pipeline_layout_info, nullptr,
      &state.compute_pipeline_layout);

  VkComputePipelineCreateInfo compute_pipeline_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = stage_info,
    .layout = state.compute_pipeline_layout
  };
  res = vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1,
      &compute_pipeline_info, nullptr, &state.compute_pipeline);
  assert(res == VK_SUCCESS);

  vkDestroyShaderModule(state.device, shader_module, nullptr);
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
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
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

void setup_buffer_state_vert_buffers(AppState& state, int buf_index) {
  BufferState& buf_state = state.buffer_states[buf_index];

  VkDeviceSize buffer_size = sizeof(vec4) * MAX_NUM_VERTICES;
  for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
    create_buffer(state.device, state.phys_device,
        buffer_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
          VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        buf_state.vert_buffers[i], buf_state.vert_buffer_mems[i]);

    VkBufferViewCreateInfo buffer_view_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
      .buffer = buf_state.vert_buffers[i],
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = 0,
      .range = VK_WHOLE_SIZE
    };
    VkResult res = vkCreateBufferView(state.device,
        &buffer_view_info, nullptr, &buf_state.vert_buffer_views[i]);
    assert(res == VK_SUCCESS);
  }
}

void setup_buffer_state_render_desc_sets(AppState& state, int buf_index) {
  BufferState& buf_state = state.buffer_states[buf_index];

  VkDescriptorSetAllocateInfo render_desc_set_alloc_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = state.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &state.render_desc_set_layout
  };
  VkResult res = vkAllocateDescriptorSets(state.device,
      &render_desc_set_alloc_info, &buf_state.render_desc_set);
  assert(res == VK_SUCCESS);

  vector<VkWriteDescriptorSet> writes;
  for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
    VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = buf_state.render_desc_set,
      .dstBinding = i,
      .dstArrayElement = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .descriptorCount = 1,
      .pTexelBufferView = &buf_state.vert_buffer_views[i],
    };
    writes.push_back(write);
  }
  vkUpdateDescriptorSets(state.device, (uint32_t) writes.size(),
      writes.data(), 0, nullptr);
}

void setup_buffer_state_compute_desc_sets(AppState& state, int buf_index) {
  BufferState& buf_state = state.buffer_states[buf_index];
  BufferState& other_buf_state = state.buffer_states[(buf_index + 1) % 2];
  
  VkDescriptorSetAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = state.desc_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &state.compute_desc_set_layout
  };
  VkResult res = vkAllocateDescriptorSets(state.device,
      &alloc_info, &buf_state.compute_desc_set);
  assert(res == VK_SUCCESS);
 
  vector<VkWriteDescriptorSet> writes;
  // the writes for the bindings to the texel buffers
  for (uint32_t i = 0; i < 2 * ATTRIBUTES_COUNT; ++i) {
    VkBufferView& buf_view = i < ATTRIBUTES_COUNT ?
      buf_state.vert_buffer_views[i] :
      other_buf_state.vert_buffer_views[i % ATTRIBUTES_COUNT];
    
    VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = buf_state.compute_desc_set,
      .dstBinding = i,
      .dstArrayElement = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .descriptorCount = 1,
      .pTexelBufferView = &buf_view,
    };
    writes.push_back(write);
  }
  // the write for the shared storage buffer
  VkDescriptorBufferInfo storage_buffer_info = {
    .buffer = state.compute_storage_buffer,
    .offset = 0,
    .range = VK_WHOLE_SIZE
  };
  VkWriteDescriptorSet compute_storage_write = {
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = buf_state.compute_desc_set,
    .dstBinding = 2 * ATTRIBUTES_COUNT,
    .dstArrayElement = 0,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1,
    .pBufferInfo = &storage_buffer_info
  };
  writes.push_back(compute_storage_write);

  vkUpdateDescriptorSets(state.device, (uint32_t) writes.size(),
      writes.data(), 0, nullptr);
}

void setup_buffer_state_desc_sets(AppState& state, int buf_index) {
  setup_buffer_state_render_desc_sets(state, buf_index);
  setup_buffer_state_compute_desc_sets(state, buf_index);
}

void setup_buffer_states(AppState& state) {
  for (int i = 0; i < state.buffer_states.size(); ++i) {
    setup_buffer_state_vert_buffers(state, i); 
  }
  for (int i = 0; i < state.buffer_states.size(); ++i) {
    setup_buffer_state_desc_sets(state, i);
  }
}

void setup_compute_storage_buffer(AppState& state) {
  VkDeviceSize buffer_size = sizeof(ComputeStorage);
  create_buffer(state.device, state.phys_device,
      buffer_size,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      state.compute_storage_buffer, state.compute_storage_buffer_mem);
}

// TODO - remove
/*
void old_setup_vertex_buffer(AppState& state, vector<Vertex>& vertices) {
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
*/

void setup_index_buffers(AppState& state) {
  for (uint32_t i = 0; i < PIPELINES_COUNT; ++i) {
    create_buffer(state.device, state.phys_device,
        sizeof(uint16_t) * MAX_NUM_INDICES,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          state.index_buffers[i], state.index_buffer_mems[i]);
  }
}

void setup_descriptor_pool(AppState& state) {
  uint32_t size = 1000;
  vector<VkDescriptorPoolSize> pool_sizes;
  for (int i = 0; i < VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; ++i) {
    VkDescriptorPoolSize pool_size = {(VkDescriptorType) i, size};
    pool_sizes.push_back(pool_size);
  }
  VkDescriptorPoolCreateInfo desc_pool_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    .poolSizeCount = (uint32_t) pool_sizes.size(),
    .pPoolSizes = pool_sizes.data(),
    .maxSets = (uint32_t) (size * pool_sizes.size())
  };
  VkResult res = vkCreateDescriptorPool(state.device, &desc_pool_info,
      nullptr, &state.desc_pool);
  assert(res == VK_SUCCESS);
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

void copy_data_to_buffer(AppState& state, StagingBuf& staging,
    void* src_data, uint32_t buffer_size, VkBuffer& dst_buffer) {
  void* staging_data;
  vkMapMemory(state.device, staging.mem,
      0, buffer_size, 0, &staging_data);
  memcpy(staging_data, src_data, (size_t) buffer_size);
  vkUnmapMemory(state.device, staging.mem);
  copy_buffer(state, staging.buf, dst_buffer, buffer_size);
}

void copy_data_from_buffer(AppState& state, StagingBuf& staging,
    void* dst_data, uint32_t buffer_size, VkBuffer& src_buffer) {
  copy_buffer(state, src_buffer, staging.buf, buffer_size);
  void* staging_data;
  vkMapMemory(state.device, staging.mem,
      0, buffer_size, 0, &staging_data);
  memcpy(dst_data, staging_data, (size_t) buffer_size);
  vkUnmapMemory(state.device, staging.mem);
}

void write_nodes_to_buffers(AppState& state, MorphNodes& node_vecs) {
  uint32_t node_count = node_vecs.pos_vec.size();
  assert(node_count < MAX_NUM_VERTICES);
  state.node_count = node_count;

  VkDeviceSize buffer_size = sizeof(vec4) * node_count;
  StagingBuf staging(state, buffer_size);
  
  // Always write to the first buffer
  BufferState& buf_state = state.buffer_states[0];

  vector<void*> copy_srcs = node_vecs.data_ptrs();
  for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
    copy_data_to_buffer(state, staging,
        copy_srcs[i], buffer_size, buf_state.vert_buffers[i]);
  }

  staging.cleanup(state);
}

MorphNodes read_nodes_from_buffers(AppState& state, uint32_t buf_index) {
  if (state.node_count == 0) {
    return MorphNodes(0);
  }
  BufferState& buf_state = state.buffer_states[buf_index];
  MorphNodes node_vecs(state.node_count);
  VkDeviceSize buffer_size = sizeof(vec4) * state.node_count;

  StagingBuf staging(state, buffer_size);
  vector<void*> copy_dsts = node_vecs.data_ptrs();
  for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
    copy_data_from_buffer(state, staging, copy_dsts[i],
        buffer_size, buf_state.vert_buffers[i]);
  }
  staging.cleanup(state);

  return node_vecs;
}

void log_nodes(MorphNodes& node_vecs) {
  printf("%lu nodes:\n", node_vecs.pos_vec.size()); 
  for (int i = 0; i < node_vecs.pos_vec.size(); ++i) {
    MorphNode node = node_vecs.node_at(i);
    printf("%4d %s\n", i, raw_node_str(node).c_str());
  }
  printf("\n\n");
}

void record_render_pass(AppState& state, uint32_t buffer_index) {
  uint32_t i = buffer_index;

  // moves the command buffer back to the initial state so that we
  // may record again
  VkResult res = vkResetCommandBuffer(state.cmd_buffers[i],
      VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
  assert(res == VK_SUCCESS);

  VkCommandBufferBeginInfo begin_info = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    .pInheritanceInfo = nullptr
  };
  res = vkBeginCommandBuffer(state.cmd_buffers[i], &begin_info);
  assert(res == VK_SUCCESS);

  array<VkClearValue, 2> clear_values = {};
  clear_values[0].color = {1.0f, 1.0f, 1.0f, 1.0f};
  clear_values[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo render_pass_info = {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    .renderPass = state.render_pass,
    .framebuffer = state.swapchain_framebuffers[i],
    .renderArea.offset = {0, 0},
    .renderArea.extent = state.target_extent,
    .clearValueCount = (uint32_t) clear_values.size(),
    .pClearValues = clear_values.data()
  };

  BufferState& buf_state = state.buffer_states[state.result_buffer];
  auto& vert_buffers = buf_state.vert_buffers;
  vector<VkDeviceSize> byte_offsets(vert_buffers.size(), 0);

  // update push constants
  Camera& cam = state.cam;
  float aspect_ratio = state.target_extent.width / (float) state.target_extent.height;
  mat4 model_mat = mat4(1.0f);
  mat4 view_mat = glm::lookAt(cam.pos(), cam.pos() + cam.forward(), cam.up());
  mat4 proj_mat = glm::perspective((float) M_PI / 4.0f, aspect_ratio, 0.1f, 10000.0f);
  // invert Y b/c vulkan's y-axis is inverted wrt OpenGL
	proj_mat[1][1] *= -1;
  RenderPushConstants push_consts(model_mat, view_mat, proj_mat,
      state.render_unifs);

  vkCmdBeginRenderPass(state.cmd_buffers[i], &render_pass_info,
        VK_SUBPASS_CONTENTS_INLINE);

  // draw the structure for each active pipeline 
  for (uint32_t pipeline_index = 0; pipeline_index < PIPELINES_COUNT; ++pipeline_index) {
    if (!state.controls.pipeline_toggles[pipeline_index] ||
        state.index_counts[pipeline_index] == 0) {
      continue;
    }
    vkCmdBindPipeline(state.cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
        state.graphics_pipelines[pipeline_index]);
    vkCmdBindVertexBuffers(state.cmd_buffers[i], 0, vert_buffers.size(),
        vert_buffers.data(), byte_offsets.data());
    vkCmdBindIndexBuffer(state.cmd_buffers[i],
        state.index_buffers[pipeline_index], 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(state.cmd_buffers[i],
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        state.render_pipeline_layout, 0, 1,
        &buf_state.render_desc_set, 0, nullptr);
    vkCmdPushConstants(state.cmd_buffers[i], state.render_pipeline_layout,
        VK_SHADER_STAGE_VERTEX_BIT, 0,
        sizeof(RenderPushConstants), &push_consts);

    // TODO - remove later
    //vkCmdDraw(state.cmd_buffers[i], state.node_count, 1, 0, 0);

    vkCmdDrawIndexed(state.cmd_buffers[i], state.index_counts[pipeline_index],
        1, 0, 0, 0);
  }

  ImGui_ImplVulkan_RenderDrawData(
      ImGui::GetDrawData(), state.cmd_buffers[i]);
    
  vkCmdEndRenderPass(state.cmd_buffers[i]);

  res = vkEndCommandBuffer(state.cmd_buffers[i]);
  assert(res == VK_SUCCESS);
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

void cleanup_swapchain(AppState& state) {
  vkDestroyImageView(state.device, state.depth_img_view, nullptr);
  vkDestroyImage(state.device, state.depth_img, nullptr);
  vkFreeMemory(state.device, state.depth_img_mem, nullptr);

  for (VkFramebuffer& fb : state.swapchain_framebuffers) {
    vkDestroyFramebuffer(state.device, fb, nullptr);
  }
  vkFreeCommandBuffers(state.device, state.cmd_pool,
      (uint32_t) state.cmd_buffers.size(), state.cmd_buffers.data());

  for (VkPipeline& pipeline : state.graphics_pipelines) {
    vkDestroyPipeline(state.device, pipeline, nullptr);
  }
  vkDestroyPipelineLayout(state.device, state.render_pipeline_layout, nullptr);

  vkDestroyRenderPass(state.device, state.render_pass, nullptr);
  for (VkImageView& img_view : state.swapchain_img_views) {
    vkDestroyImageView(state.device, img_view, nullptr);
  }
  // TODO this triggers a segfault, bug with MoltenVK:
  // https://github.com/KhronosGroup/MoltenVK/issues/584
  // Validation layer will complain for now
  //vkDestroySwapchainKHR(state.device, state.swapchain, nullptr);
}

void cleanup_vulkan(AppState& state) {
  cleanup_swapchain(state);

  // cleanup the buffer states
  for (BufferState& buf_state : state.buffer_states) {
    for (uint32_t i = 0; i < ATTRIBUTES_COUNT; ++i) {
      vkDestroyBufferView(state.device,
          buf_state.vert_buffer_views[i], nullptr);
      vkDestroyBuffer(state.device,
          buf_state.vert_buffers[i], nullptr);
      vkFreeMemory(state.device,
          buf_state.vert_buffer_mems[i], nullptr);
    }
  }
  vkDestroyPipeline(state.device, state.compute_pipeline, nullptr);
  vkDestroyPipelineLayout(state.device, state.compute_pipeline_layout, nullptr);

  vkDestroyDescriptorPool(state.device, state.desc_pool, nullptr);
  
  vkDestroyDescriptorSetLayout(state.device,
      state.render_desc_set_layout, nullptr);
  vkDestroyDescriptorSetLayout(state.device,
      state.compute_desc_set_layout, nullptr);

  for (uint32_t i = 0; i < PIPELINES_COUNT; ++i) {
    vkDestroyBuffer(state.device, state.index_buffers[i], nullptr);
    vkFreeMemory(state.device, state.index_buffer_mems[i], nullptr);
  }

  for (int i = 0; i < max_frames_in_flight; ++i) {
    vkDestroySemaphore(state.device, state.render_done_semas[i], nullptr);
    vkDestroySemaphore(state.device, state.img_available_semas[i], nullptr);
    vkDestroyFence(state.device, state.in_flight_fences[i], nullptr);
  }
  vkDestroyCommandPool(state.device, state.cmd_pool, nullptr);

  vkDestroyDevice(state.device, nullptr);

  state.destroy_debug_utils(state.inst, state.debug_messenger, nullptr);

  vkDestroySurfaceKHR(state.inst, state.surface, nullptr);
  vkDestroyInstance(state.inst, nullptr);
}

void reload_programs(AppState& state) {
  vkDeviceWaitIdle(state.device);

  // recreate graphics and compute pipelines

  for (VkPipeline& pipeline : state.graphics_pipelines) {
    vkDestroyPipeline(state.device, pipeline, nullptr);
  }
  vkDestroyPipelineLayout(state.device,
      state.render_pipeline_layout, nullptr);
  setup_graphics_pipelines(state);

  vkDestroyPipeline(state.device, state.compute_pipeline, nullptr);
  vkDestroyPipelineLayout(state.device,
      state.compute_pipeline_layout, nullptr);
  setup_compute_pipeline(state);
}

void recreate_swapchain(AppState& state) {
  // if the window is minimized, wait until it comes to the foreground
  // again
  int fb_w = 0;
  int fb_h = 0;
  while (fb_w == 0 || fb_h == 0) {
    glfwGetFramebufferSize(state.win, &fb_w, &fb_h);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(state.device);
  cleanup_swapchain(state);

  setup_swapchain(state);
  setup_renderpass(state);
  setup_graphics_pipelines(state);
  setup_depth_resources(state);
  setup_framebuffers(state);
  setup_command_buffers(state);
  ImGui_ImplVulkan_SetMinImageCount(state.surface_caps.minImageCount);
}

void init_vulkan(AppState& state) {

  setup_vertex_attr_desc(state);
  setup_instance(state);
  setup_debug_callback(state);
  setup_surface(state); 
  setup_physical_device(state);
  setup_logical_device(state);
  setup_swapchain(state);
  setup_command_pool(state);
  setup_depth_resources(state);
  setup_index_buffers(state);

  setup_desc_set_layouts(state);

  setup_renderpass(state);
  setup_framebuffers(state);
  setup_graphics_pipelines(state);
  setup_compute_pipeline(state);

  setup_descriptor_pool(state);
  setup_compute_storage_buffer(state);
  setup_buffer_states(state);

  setup_command_buffers(state);
  setup_sync_objects(state);
}

void render_frame(AppState& state) {
  size_t current_frame = state.current_frame;

  vkWaitForFences(state.device, 1, &state.in_flight_fences[current_frame],
        VK_TRUE, std::numeric_limits<uint64_t>::max());
  
  uint32_t img_index;
  VkResult res = vkAcquireNextImageKHR(state.device, state.swapchain,
      std::numeric_limits<uint64_t>::max(),
      state.img_available_semas[current_frame],
      VK_NULL_HANDLE, &img_index);
  if (res == VK_ERROR_OUT_OF_DATE_KHR || state.framebuffer_resized) {
    state.framebuffer_resized = false;
    recreate_swapchain(state);
    return;
  } 
    
  record_render_pass(state, img_index);

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
  vkResetFences(state.device, 1, &state.in_flight_fences[current_frame]);
  res = vkQueueSubmit(state.queue, 1, &submit_info,
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
  res = vkQueuePresentKHR(state.queue, &present_info);
  if (res == VK_ERROR_OUT_OF_DATE_KHR) {
    recreate_swapchain(state);
  }

  state.current_frame = (current_frame + 1) % max_frames_in_flight;
}

void run_debug_test(AppState& state) {
  printf("running debug test\n");
  vector<MorphNode> nodes = {
    MorphNode(vec4(0.0f), vec4(0.0f), vec4(0.0f), vec4(0.0f)),
    MorphNode(vec4(0.0f), vec4(1.0f), vec4(2.0f), vec4(3.0f)),
  };
  MorphNodes in_node_vecs(nodes);
  printf("in nodes:\n");
  log_nodes(in_node_vecs);
  printf("\n");

  write_nodes_to_buffers(state, in_node_vecs);

  MorphNodes out_node_vecs = read_nodes_from_buffers(
      state, state.result_buffer);
  printf("out nodes:\n");
  log_nodes(out_node_vecs);
}

void log_buffers(AppState& state) {
  for (uint32_t i = 0; i < state.buffer_states.size(); ++i) {
    MorphNodes out_node_vecs = read_nodes_from_buffers(state, i);
    printf("buffer %d:\n", i);
    log_nodes(out_node_vecs);
  }
}

vec3 gen_sphere(vec2 unit) {
  float v_angle = unit[1] * M_PI;
  float h_angle = unit[0] * 2.0 * M_PI;
  return vec3(
      sin(v_angle) * cos(h_angle),
      sin(v_angle) * sin(h_angle),
      cos(v_angle));
}

vec3 gen_square(vec2 unit) {
  return vec3(unit, 0);
}

vec3 gen_plane(vec2 unit) {
  vec2 plane_pos = 10.0f * (unit - 0.5f);
  return vec3(plane_pos[0], 0.0f, plane_pos[1]);
}

// Helper for gen_morph_data
// returns -1 if the coord is outside the plane
int coord_to_index(ivec2 coord, ivec2 samples) {
  if ((0 <= coord[0] && coord[0] < samples[0]) &&
      (0 <= coord[1] && coord[1] < samples[1])) {
    return coord[0] % samples[0] + samples[0] * (coord[1] % samples[1]);
  } else {
    return -1;
  }
}

// Outputs the nodes, for rendering
void gen_morph_data(ivec2 samples, vector<MorphNode>& out_nodes) {
  vector<MorphNode> vertex_nodes;
  vertex_nodes.reserve(samples[0] * samples[1]);
  for (int y = 0; y < samples[1]; ++y) {
    for (int x = 0; x < samples[0]; ++x) {
      ivec2 coord(x, y);
      vec3 pos = gen_plane(vec2(coord) / vec2(samples - 1));

      int upper_neighbor = coord_to_index(coord + ivec2(0, 1), samples);
      int lower_neighbor = coord_to_index(coord + ivec2(0, -1), samples);
      int right_neighbor = coord_to_index(coord + ivec2(1, 0), samples);
      int left_neighbor = coord_to_index(coord + ivec2(-1, 0), samples);
      vec4 neighbors(
          (float) right_neighbor, (float) upper_neighbor,
          (float) left_neighbor, (float) lower_neighbor);

      MorphNode vert_node(vec4(pos, 0.0), vec4(0.0), neighbors, vec4(0.0));
      vertex_nodes.push_back(vert_node);
    }
  }
  out_nodes = std::move(vertex_nodes);
}

using IndexPair = std::pair<uint16_t, uint16_t>;

struct IndexPairHash {
	size_t operator()(const IndexPair& pair) const {
    size_t seed = 0;
    hash_combine(seed, pair.first);
    hash_combine(seed, pair.second);
    return seed;
	}
};

vector<uint16_t> gen_triangle_indices(AppState& state, MorphNodes& node_vecs) {
  // TODO - faces is harder than one might think, defer
  // Points and lines is sufficient for visualization
  vector<uint16_t> indices;
  return indices;
}

vector<uint16_t> gen_line_indices(AppState& state, MorphNodes& node_vecs) {

  // add a line for every valid edge
  // use a set of edges to prevent duplicates
  vector<uint16_t> indices;
  uint32_t node_count = node_vecs.pos_vec.size();
  unordered_set<IndexPair, IndexPairHash> edge_set;
  for (uint32_t i = 0; i < node_count; ++i) {
    vec4 neighbors = node_vecs.neighbors_vec[i];  
    for (int j = 0; j < 4; ++j) {
      int n_index = neighbors[j];
      if (n_index == -1.0) {
        continue;
      }
      IndexPair pair = i < n_index ?
        IndexPair(i, n_index) : IndexPair(n_index, i);
      if (edge_set.find(pair) == edge_set.end()) {
        edge_set.insert(pair);
        indices.push_back(pair.first);
        indices.push_back(pair.second);
      }
    }
  }
  return indices;
}

vector<uint16_t> gen_point_indices(AppState& state, MorphNodes& node_vecs) {
  uint32_t node_count = node_vecs.pos_vec.size();
  vector<uint16_t> indices(node_count);
  for (uint32_t i = 0; i < node_count; ++i) {
    indices[i] = i;
  }
  return indices;
}

void update_indices(AppState& state, MorphNodes& node_vecs) {
  array<vector<uint16_t>, PIPELINES_COUNT> pipeline_indices = {
    gen_point_indices(state, node_vecs),
    gen_line_indices(state, node_vecs),
    gen_triangle_indices(state, node_vecs)
  };

  // debug logging
  vector<tuple<const char*, bool, int>> log_toggles = {
    {"point", state.controls.log_point_indices, 1},
    {"line", state.controls.log_line_indices, 2},
    {"triangle", state.controls.log_triangle_indices, 3}
  };
  for (uint32_t p_i = 0; p_i < PIPELINES_COUNT; ++p_i) {
    const char* label = get<0>(log_toggles[p_i]);
    bool should_print = get<1>(log_toggles[p_i]);
    int entries_per_line = get<2>(log_toggles[p_i]);
    if (!should_print) {
      continue;
    }
    vector<uint16_t>& indices = pipeline_indices[p_i];
    printf("\n%s indices (%lu):\n", label, indices.size());
    for (int i = 0; i < indices.size(); ++i) {
      printf("%s%4d ",
          i % entries_per_line == 0 ? "\n" : "", indices[i]);
    }
    printf("\n");
  }

  // copy the indices to their respective buffers
  for (uint32_t i = 0; i < PIPELINES_COUNT; ++i) {
    vector<uint16_t>& indices = pipeline_indices[i];
    if (indices.size() == 0) {
      continue;
    }
    assert(indices.size() < MAX_NUM_INDICES);
    state.index_counts[i] = indices.size();
    VkDeviceSize buffer_size = sizeof(indices[0]) * indices.size();

    StagingBuf staging(state, buffer_size);
    copy_data_to_buffer(state, staging, indices.data(),
        buffer_size, state.index_buffers[i]);
    staging.cleanup(state);
  }
}

void set_initial_sim_data(AppState& state) {

  ivec2 zygote_samples(state.controls.num_zygote_samples);
  vector<MorphNode> nodes;
  gen_morph_data(zygote_samples, nodes);
  MorphNodes node_vecs(nodes);
 
  if (state.controls.log_input_nodes) {
    printf("input nodes:\n");
    log_nodes(node_vecs);
  }

  write_nodes_to_buffers(state, node_vecs);
}

void log_compute_storage(ComputeStorage& cs) {
  printf(
      "ctr0 %4d, ctr1 %4d\n"
      "start0 %4d, end0 %4d\n"
      "start1 %4d, end1 %4d\n",
      cs.step_counters[0], cs.step_counters[1],
      cs.start_ptrs[0], cs.end_ptrs[0],
      cs.start_ptrs[1], cs.end_ptrs[1]);
  printf("queue mem:\n");
  uint32_t q_len = cs.queue_mem.size();
  for (uint32_t i = 0; i < cs.queue_mem.size(); ++i) {
    printf("(%2s %2s %2s %2s) %4d: %4d\n",
        cs.start_ptrs[0] % q_len == i ? "s0" : "",
        cs.end_ptrs[0] % q_len == i ? "e0" : "",
        cs.start_ptrs[1] % q_len == i ? "s1" : "",
        cs.end_ptrs[1] % q_len == i ? "e1" : "",
        i, cs.queue_mem[i]);
  }
}

void write_to_compute_storage(AppState& state,
    ComputeStorage& compute_storage) {
  StagingBuf staging(state, sizeof(ComputeStorage));
  copy_data_to_buffer(state, staging, &compute_storage,
      sizeof(compute_storage), state.compute_storage_buffer);
  staging.cleanup(state);
}

ComputeStorage read_from_compute_storage(AppState& state) {
  ComputeStorage compute_storage;
  StagingBuf staging(state, sizeof(compute_storage));
  copy_data_from_buffer(state, staging, &compute_storage,
      sizeof(compute_storage), state.compute_storage_buffer);
  staging.cleanup(state);
  return compute_storage;
}

void setup_test_queue(ComputeStorage& cs) {
  cs.queue_mem = {1, 2, 3};
  cs.start_ptrs = {0, 0};
  cs.end_ptrs = {3, 3};
}

void dispatch_simulation(AppState& state) {

  // init shared storage
  ComputeStorage in_compute_storage;
  setup_test_queue(in_compute_storage);
  write_to_compute_storage(state, in_compute_storage);
 
  VkCommandBuffer tmp_buffer = begin_single_time_commands(state);

  VkMemoryBarrier mem_barrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
  };

  vkCmdBindPipeline(tmp_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      state.compute_pipeline);
  
  uint32_t num_iters = state.controls.num_iters;
  for (uint32_t i = 0; i < num_iters; ++i) {
    BufferState& cur_buf = state.buffer_states[i & 1];

    if (i > 0) {
      // require that the previous iter writes are available to
      // this iteration's reads
      vkCmdPipelineBarrier(tmp_buffer,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
          1, &mem_barrier,
          0, nullptr,
          0, nullptr);
    }

    // TODO - are these bindings read at the time that the dispatch is
    // recorded, or when it executes? Makes massive difference

    vkCmdBindDescriptorSets(tmp_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        state.compute_pipeline_layout, 0, 1, &cur_buf.compute_desc_set,
        0, nullptr);

    ComputePushConstants push_consts(
        state.node_count, i, STORAGE_QUEUE_LEN,
        state.compute_unifs);
    vkCmdPushConstants(tmp_buffer, state.compute_pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, 
        sizeof(ComputePushConstants), &push_consts);

    uint32_t groups_x = state.node_count / LOCAL_WORKGROUP_SIZE + 1;
    vkCmdDispatch(tmp_buffer, groups_x, 1, 1);
  }
  state.result_buffer = num_iters & 1;

  end_single_time_commands(state, tmp_buffer);

  // debug logging

  if (state.controls.log_input_compute_storage) {
    printf("input compute storage:\n");
    log_compute_storage(in_compute_storage);
  }
  if (state.controls.log_output_compute_storage) {
    printf("output compute storage:\n");
    ComputeStorage out_compute_storage =
      read_from_compute_storage(state);
    log_compute_storage(out_compute_storage);
  }
}

void run_simulation_pipeline(AppState& state) { 
  set_initial_sim_data(state);
  dispatch_simulation(state);

  MorphNodes node_vecs = read_nodes_from_buffers(
      state, state.result_buffer);

  update_indices(state, node_vecs);

  if (state.controls.log_output_nodes) {
    printf("output nodes:\n");
    log_nodes(node_vecs);
  }
}

void framebuffer_resize_callback(GLFWwindow* win,
    int w, int h) {
  AppState* state = reinterpret_cast<AppState*>(
      glfwGetWindowUserPointer(win));
  state->framebuffer_resized = true;
}

void handle_key_event(GLFWwindow* win, int key, int scancode,
    int action, int mods) {
	AppState* state = reinterpret_cast<AppState*>(
      glfwGetWindowUserPointer(win));
  Controls& controls = state->controls;

  if (key == GLFW_KEY_R && action == GLFW_PRESS) {
    // reset camera pos
    state->cam = Camera();
  }
  if (key == GLFW_KEY_F && action == GLFW_PRESS) {
		controls.cam_spherical_mode = !controls.cam_spherical_mode;
  }
  if (key == GLFW_KEY_P && action == GLFW_PRESS) {
    // TODO - also recreate the compute pipeline
    printf("reloading program\n");
    reload_programs(*state);
  }
  if (key == GLFW_KEY_C && action == GLFW_PRESS) {
    run_simulation_pipeline(*state);  
  }
  // TODO - only for debugging
  if (key == GLFW_KEY_N && action == GLFW_PRESS) {
    run_debug_test(*state);
  }
}

void init_glfw(AppState& state) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    throw std::runtime_error("Error on glfwInit()");
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  state.win = glfwCreateWindow(800, 600, "morph",
      nullptr, nullptr);
  assert(glfwVulkanSupported() == GLFW_TRUE);
  glfwSetWindowUserPointer(state.win, &state);
  glfwSetFramebufferSizeCallback(state.win,
      framebuffer_resize_callback);
  glfwSetKeyCallback(state.win, handle_key_event);
}

void gen_user_uniforms_ui(vector<UserUnif>& user_unifs) {
  bool should_reset_unifs = ImGui::Button("restore defaults");
  for (UserUnif& user_unif : user_unifs) {
    if (should_reset_unifs) {
      user_unif.current_val = user_unif.default_val;
    }
    ImGui::DragScalarN(user_unif.name.c_str(), ImGuiDataType_Float,
        &user_unif.current_val[0], user_unif.num_comps, user_unif.drag_speed,
        &user_unif.min_val, &user_unif.max_val, "%.3f");
  }
}

void upload_imgui_fonts(AppState& state) {
  VkCommandBuffer tmp_buffer = begin_single_time_commands(state);
  ImGui_ImplVulkan_CreateFontsTexture(tmp_buffer);
  end_single_time_commands(state, tmp_buffer);
  ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void update_camera_cartesian(AppState& state) {
  GLFWwindow* win = state.win;
  Controls& controls = state.controls;
  Camera& cam = state.cam;

  vec3 delta(0.0);
  if (glfwGetKey(win, GLFW_KEY_W)) {
    delta += vec3(0.0,0.0,1.0);
  }
  if (glfwGetKey(win, GLFW_KEY_A)) {
    delta += vec3(1.0,0.0,0.0);
  }
  if (glfwGetKey(win, GLFW_KEY_S)) {
    delta += vec3(0.0,0.0,-1.0);
  }
  if (glfwGetKey(win, GLFW_KEY_D)) {
    delta += vec3(-1.0,0.0,0.0);
  }
  if (glfwGetKey(win, GLFW_KEY_Q)) {
    delta += vec3(0.0,-1.0,0.0);
  }
  if (glfwGetKey(win, GLFW_KEY_E)) {
    delta += vec3(0.0,1.0,0.0);
  }
  float delta_secs = 1.0f / controls.target_fps;
  vec3 trans = delta * 20.0f * delta_secs;
  mat4 trans_mat = glm::translate(mat4(1.0), trans);

  mat4 rot_mat(1.0);
  float amt_degs = delta_secs * 5.0f;
  if (glfwGetKey(win, GLFW_KEY_LEFT)) {
    rot_mat = glm::rotate(mat4(1.0), amt_degs, vec3(0.0,1.0,0.0));
  } else if (glfwGetKey(win, GLFW_KEY_RIGHT)) {
    rot_mat = glm::rotate(mat4(1.0), -amt_degs, vec3(0.0,1.0,0.0));
  } else if (glfwGetKey(win, GLFW_KEY_UP)) {
    rot_mat = glm::rotate(mat4(1.0), amt_degs, vec3(1.0,0.0,0.0));
  } else if (glfwGetKey(win, GLFW_KEY_DOWN)) {
    rot_mat = glm::rotate(mat4(1.0), -amt_degs, vec3(1.0,0.0,0.0));
  }
  
  cam.cam_to_world = cam.cam_to_world * rot_mat * trans_mat;
}

void update_camera_spherical(AppState& state) {
  GLFWwindow* win = state.win;
  Controls& controls = state.controls;
  Camera& cam = state.cam;

  vec3 eye = cam.pos();
  float cur_r = length(eye);
  float h_angle = atan2(eye.z, eye.x);
  float v_angle = atan2(length(vec2(eye.x, eye.z)), eye.y);

  float delta_r = 0.0;
  float delta_h_angle = 0.0;
  float delta_v_angle = 0.0;
  if (glfwGetKey(win, GLFW_KEY_W)) {
    delta_r = -1.0;
  }
  if (glfwGetKey(win, GLFW_KEY_A)) {
    delta_h_angle = 1.0;
  }
  if (glfwGetKey(win, GLFW_KEY_S)) {
    delta_r = 1.0;
  }
  if (glfwGetKey(win, GLFW_KEY_D)) {
    delta_h_angle = -1.0;
  }
  if (glfwGetKey(win, GLFW_KEY_Q)) {
    delta_v_angle = 1.0;
  }
  if (glfwGetKey(win, GLFW_KEY_E)) {
    delta_v_angle = -1.0;
  }
  float delta_secs = 1.0f / controls.target_fps;
  float new_r = cur_r + 30.0f * delta_r * delta_secs;
  float new_h_angle = h_angle + M_PI / 2.0 * delta_h_angle * delta_secs;
  float new_v_angle = v_angle + M_PI / 2.0 * delta_v_angle * delta_secs;

  vec3 pos = new_r * vec3(
      cos(new_h_angle) * sin(new_v_angle),
      cos(new_v_angle),
      sin(new_h_angle) * sin(new_v_angle)
      );
  cam.set_view(pos, vec3(0.0));
}

void update_camera(AppState& state) {
  if (state.controls.cam_spherical_mode) {
    update_camera_spherical(state);
  } else {
    update_camera_cartesian(state);
  }
}

void create_ui(AppState& state) {
  Controls& controls = state.controls;

  ImGui::Begin("dev console", &controls.show_dev_console);

  ImGui::Separator();
  ImGui::Text("camera:");
  ImGui::Text("eye: %s", vec3_str(state.cam.pos()).c_str());
  ImGui::Text("forward: %s", vec3_str(state.cam.forward()).c_str());
  ImGui::Text("mode: %s", controls.cam_spherical_mode ? "spherical" : "cartesian");

  ImGui::Separator();
  ImGui::Text("render controls:");
  array<const char*, PIPELINES_COUNT> pipeline_names = {
    "points", "lines", "triangles"
  };
  for (uint32_t i = 0; i < PIPELINES_COUNT; ++i) {
    ImGui::Checkbox(pipeline_names[i], &controls.pipeline_toggles[i]);
  }

  ImGui::Separator();
  ImGui::Text("render program controls:");
  ImGui::PushID("render");
  gen_user_uniforms_ui(state.render_unifs);
  ImGui::PopID();

  ImGui::Separator();
  ImGui::Text("compute program controls:");
  ImGui::PushID("compute");
  gen_user_uniforms_ui(state.compute_unifs);
  ImGui::PopID();

  ImGui::Separator();
  ImGui::Text("simulation controls:");
  ImGui::Text("init data:");
  ImGui::InputInt("AxA samples", &controls.num_zygote_samples);
  controls.num_zygote_samples = clamp(
      controls.num_zygote_samples, 2, 500);
  
  ImGui::Text("simulation:"); 
  int max_iter_num = 1*1000*1000*1000;
  ImGui::DragInt("iter num", &controls.num_iters, 0.2f, 0, max_iter_num);
  if (ImGui::Button("run once")) {
    run_simulation_pipeline(state);  
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
    run_simulation_pipeline(state);
  }

  vector<pair<string, bool*>> log_controls = {
    {"input nodes", &controls.log_input_nodes},
    {"output nodes", &controls.log_output_nodes},
    {"point indices", &controls.log_point_indices},
    {"line indices", &controls.log_line_indices},
    {"triangle indices", &controls.log_triangle_indices},
    {"input compute storage", &controls.log_input_compute_storage},
    {"output compute storage", &controls.log_output_compute_storage},
    {"durations", &controls.log_durations},
  };
  ImGui::Separator();
  ImGui::Text("debug");
  ImGui::Text("Note that logging will not occur while animating");
  for (auto& log_con : log_controls) {
    string log_msg = string("log ") + log_con.first;
    ImGui::Checkbox(log_msg.c_str(), log_con.second);

    // turn off logging while animating, the IO becomes a bottleneck
    if (controls.animating_sim) {
      *log_con.second = false;
    }
  }
  if (ImGui::Button("log buffers")) {
    log_buffers(state);
  }

  ImGui::Separator();
  ImGui::Text("instructions:");
  ImGui::Text("%s", INSTRUCTIONS_STRING);

  ImGui::End();
}

void main_loop(AppState& state) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForVulkan(state.win, true);
  ImGui_ImplVulkan_InitInfo init_info = {
    .Instance = state.inst,
    .PhysicalDevice = state.phys_device,
    .Device = state.device,
    .QueueFamily = state.target_family_index,
    .Queue = state.queue,
    .PipelineCache = VK_NULL_HANDLE,
    .DescriptorPool = state.desc_pool,
    .Allocator = nullptr,
    .MinImageCount = state.surface_caps.minImageCount,
    .ImageCount = state.target_image_count,
    .CheckVkResultFn = check_vk_result
  };
  ImGui_ImplVulkan_Init(&init_info, state.render_pass);
  upload_imgui_fonts(state);

  state.current_frame = 0;
  while (!glfwWindowShouldClose(state.win)) {
    glfwPollEvents();

    update_camera(state);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    create_ui(state);
    ImGui::Render();

    render_frame(state);
  }
  vkDeviceWaitIdle(state.device);

  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void cleanup_state(AppState& state) {
  cleanup_vulkan(state);

  glfwDestroyWindow(state.win);
  glfwTerminate();
}

void run_app(int argc, char** argv) {
  signal(SIGSEGV, handle_segfault);

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
