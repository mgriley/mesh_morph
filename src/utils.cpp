#include "utils.h"

const char* INSTRUCTIONS_STRING = R"--(
quickstart:
Drag the "iter num" slider to see the mesh update in real-time.

camera:
WASDEQ to move
F to switch between cartesian and spherical movement
R to reset to original position

general:
P: reload programs (check stdout for errors)
C: run simulation once (can also be done via the button)

render program controls:
render_mode:
0: boring solid color
1: color by heat
2: color by heat generation amt
3: highlight source nodes

morph program controls:
You'll need to consult shaders/growth.glsl for the particular usage
of each uniform. The names are generally sensible.

simulation pane:
Running once runs the simulation for the desired number of frames, and
the result is rendered. The initial data is a grid of AxA vertices. Many
of the morph program tunable parameters default to values that work well
with about 100x100.

animation pane:
The app starts with the animation playing, but with a speed of 0. This regenerates
the mesh every frame but does not change the iteration num automatically.
To run the simulation forwards or backwards, change the "delta iters per frame" to +/-1.
While animating, the app runs "iter num" iterations every frame, so it may be slow.

)--";

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
  sprintf(s.data(), "[%5.2f %5.2f %5.2f %5.2f]", v[0], v[1], v[2], v[3]);
  return string(s.data());
}

string ivec4_str(ivec4 v) {
  array<char, 100> s;
  sprintf(s.data(), "[%4d %4d %4d %4d]", v[0], v[1], v[2], v[3]); 
  return string(s.data());
}

