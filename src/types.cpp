#include "types.h"

UserUnif::UserUnif(string name, int num_comps, vec4 default_val,
    float min_val, float max_val, float drag_speed) :
  name(name),
  num_comps(num_comps),
  default_val(default_val),
  current_val(default_val),
  min_val(min_val),
  max_val(max_val),
  drag_speed(drag_speed)
{
}

AppState::AppState()
{
}

