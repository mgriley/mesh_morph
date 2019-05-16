#pragma once

#include <vector>
#include <array>
#include <string>

#include <cstddef>
#include <cstdlib>
#include <execinfo.h>
#include <stdio.h>
#include <unistd.h>

// Note: must include vulkan before GLFW
//#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
// used to meet vulkan alignment requirements when memcpy-ing
// struct data to GPU buffer memory
// Note that it may break down for nested structs
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// Use 0.0 to 1.0 instead of -1.0 to 1.0 for the perspective proj matrix
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/string_cast.hpp"

using namespace std;
using namespace glm;

void print_backtrace();
void handle_segfault(int sig_num);
void glfw_error_callback(int error, const char* description);

string vec3_str(vec3 v);
string vec4_str(vec4 v);
string ivec4_str(ivec4 v);

extern const char* INSTRUCTIONS_STRING;

// Taken from:
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

