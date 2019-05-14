#include "utils.h"

void print_backtrace() {
  array<void*, 15> frames{};
  int num_frames = backtrace(frames.data(), frames.size());
    backtrace_symbols_fd(frames.data(), num_frames, STDERR_FILENO);
}

void handle_segfault(int sig_num) {
  fprintf(stdout, "SEGFAULT to stdout");
  fprintf(stderr, "SEGFAULT signal: %d\n", sig_num);
  print_backtrace();
  exit(1);
}

void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

string vec3_str(vec3 v) {
  array<char, 100> s;
  sprintf(s.data(), "[%5.2f, %5.2f, %5.2f]", v[0], v[1], v[2]);
  return string(s.data());
}

string vec4_str(vec4 v) {
  array<char, 100> s;
  sprintf(s.data(), "[%5.2f, %5.2f, %5.2f, %5.2f]", v[0], v[1], v[2], v[3]);
  return string(s.data());
}

string ivec4_str(ivec4 v) {
  array<char, 100> s;
  sprintf(s.data(), "[%4d, %4d, %4d, %4d]", v[0], v[1], v[2], v[3]); 
  return string(s.data());
}

