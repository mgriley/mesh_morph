cmake_minimum_required (VERSION 3.11.0)
project (MeshMorph)
# NB: compile with debug symbols
set(CMAKE_CXX_FLAGS "-std=c++0x -g") 

# Convenience variables
# CDIR is the directory of the CMakeLists.txt file
set(CDIR ${CMAKE_CURRENT_SOURCE_DIR})
# BDIR is the build/ directory
set(BDIR ${CMAKE_CURRENT_BINARY_DIR})

set(VULKAN_PATH ${CDIR}/../vulkansdk-macos-1.1.106.0/macOS)
set(ENV{VULKAN_SDK} ${VULKAN_PATH}) 

find_package(glfw3 3.3 REQUIRED)

find_package(vulkan REQUIRED)
if (NOT Vulkan_FOUND)
  message("vulkan not found")
endif()

# compile ImGui to a static lib
set(IMGUI "${CDIR}/third_party/imgui")
file(GLOB IMGUI_TOPLEVEL_SOURCES "${IMGUI}/*.cpp")
list(APPEND IMGUI_SOURCES
  ${IMGUI_TOPLEVEL_SOURCES}
  "${IMGUI}/examples/imgui_impl_glfw.cpp"
  "${IMGUI}/examples/imgui_impl_vulkan.cpp")
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC
  ${IMGUI} "${IMGUI}/examples" include ${Vulkan_INCLUDE_DIRS})

# compile the app

set(DRIVER "${CDIR}/src/main.cpp")
file(GLOB SOURCES "src/*.cpp" "src/*.c")
list(REMOVE_ITEM SOURCES ${DRIVER})
add_library(main_lib STATIC ${SOURCES})
target_include_directories(main_lib PUBLIC include)
target_link_libraries(main_lib PUBLIC glfw Vulkan::Vulkan imgui) 
# pass the manifest file locations to the exec instead of
# specifying them on the command-line every time
target_compile_definitions(main_lib PUBLIC
  ENV_VK_ICD_FILENAMES="VK_ICD_FILENAMES=${VULKAN_PATH}/etc/vulkan/icd.d/MoltenVK_icd.json")
target_compile_definitions(main_lib PUBLIC
  ENV_VK_LAYER_PATH="VK_LAYER_PATH=${VULKAN_PATH}/etc/vulkan/explicit_layer.d")

add_executable(main_exec ${DRIVER})
target_link_libraries(main_exec PUBLIC main_lib)

