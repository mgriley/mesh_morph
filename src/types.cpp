#include "types.h"

Camera::Camera()
{
  set_view(vec3(0.0,30.0,-30.0), vec3(0.0));
}

void Camera::set_view(vec3 eye, vec3 target) {
  vec3 forward = normalize(target - eye);
  vec3 right = normalize(cross(vec3(0.0,1.0,0.0), forward));
  vec3 up = cross(forward, right);
  
  cam_to_world[0] = vec4(right, 0.0);
  cam_to_world[1] = vec4(up, 0.0);
  cam_to_world[2] = vec4(forward, 0.0);
  cam_to_world[3] = vec4(eye, 1.0);
}

vec3 Camera::pos() const {
  return vec3(cam_to_world[3]);
}

vec3 Camera::right() const {
  return vec3(cam_to_world[0]);
}

vec3 Camera::up() const {
  return vec3(cam_to_world[1]);
}

vec3 Camera::forward() const {
  return vec3(cam_to_world[2]);
}

Controls::Controls()
{
}

MorphNode::MorphNode():
  pos(0.0),
  vel(0.0),
  neighbors(-1.0),
  data(0.0),
  top_data(0.0)
{
}

MorphNode::MorphNode(vec4 pos, vec4 vel,
    vec4 neighbors, vec4 data, vec4 top_data) :
  pos(pos),
  vel(vel),
  neighbors(neighbors),
  data(data),
  top_data(top_data)
{
}

MorphNodes::MorphNodes(size_t num_nodes) :
  pos_vec(num_nodes),
  vel_vec(num_nodes),
  neighbors_vec(num_nodes),
  data_vec(num_nodes),
  top_data_vec(num_nodes)
{
}

MorphNodes::MorphNodes(vector<MorphNode> const& nodes) :
  pos_vec(nodes.size()),
  vel_vec(nodes.size()),
  neighbors_vec(nodes.size()),
  data_vec(nodes.size()),
  top_data_vec(nodes.size())
{
  for (int i = 0; i < nodes.size(); ++i) {
    MorphNode const& node = nodes[i];
    pos_vec[i] = node.pos;
    vel_vec[i] = node.vel;
    neighbors_vec[i] = node.neighbors;
    data_vec[i] = node.data;
    top_data_vec[i] = node.top_data;
  }
}

MorphNode MorphNodes::node_at(size_t i) const {
  return MorphNode(pos_vec[i], vel_vec[i], 
      neighbors_vec[i], data_vec[i], top_data_vec[i]);
}

array<void*, ATTRIBUTES_COUNT> MorphNodes::data_ptrs() { 
  return {
    pos_vec.data(), vel_vec.data(), neighbors_vec.data(),
    data_vec.data(), top_data_vec.data()
  };
}

string raw_node_str(MorphNode const& node) {
  array<char, 200> s;
  sprintf(s.data(),
      "p%s v%s n%s d%s "
      "t_d%s",
        vec4_str(node.pos).c_str(),
        vec4_str(node.vel).c_str(),
        vec4_str(node.neighbors).c_str(),
        vec4_str(node.data).c_str(),
        vec4_str(node.top_data).c_str());
  return string(s.data());
}

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

void set_user_unif_vals(const vector<UserUnif>& user_unifs,
    array<vec4, MAX_NUM_USER_UNIFS>& user_unif_vals) {
  assert(user_unifs.size() < user_unif_vals.size());
  for (int i = 0; i < user_unifs.size(); ++i) {
    user_unif_vals[i] = user_unifs[i].current_val;
  }
}

ComputeStorage::ComputeStorage()
{
}

RenderPushConstants::RenderPushConstants(mat4 model, mat4 view,
    mat4 proj, const vector<UserUnif>& user_unifs) :
  model(model), view(view), proj(proj)
{
  set_user_unif_vals(user_unifs, user_unif_vals);  
}

ComputePushConstants::ComputePushConstants(
    uint32_t node_count, uint32_t inactive_node_count,
    uint32_t iter_num, uint32_t queue_len,
    const vector<UserUnif>& user_unifs) :
  node_count(node_count), inactive_node_count(inactive_node_count),
  iter_num(iter_num), queue_len(queue_len)
{
  set_user_unif_vals(user_unifs, user_unif_vals);  
}

BufferState::BufferState()
{
}

AppState::AppState()
{
}


