#include <torch/extension.h>    // Pybind11 포함
// #include <ATen/ATen.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <map>
#include <string>

using namespace std;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// CUDA declarations
// torch::Tensor normalize_cuda(torch::Tensor ix, torch::Tensor iy, const int IW, const int IH);
torch::Tensor normalize_cuda(torch::Tensor coords, torch::Tensor IW, torch::Tensor IHs);
torch::Tensor normalize_offset_cuda(torch::Tensor input, int len, torch::Tensor offset);
torch::Tensor get_corner_cuda(torch::Tensor coord);
torch::Tensor get_weight_cuda(torch::Tensor coord, torch::Tensor corner);
torch::Tensor get_point_cuda(torch::Tensor weight);
// torch::Tensor gather_cuda(std::vector<torch::Tensor> input, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
// torch::Tensor gather_cuda(std::vector<torch::Tensor> input, torch::Tensor corner, torch::Tensor IW, 
// torch::Tensor book, const int C);
torch::Tensor gather_cuda(torch::Tensor block_x, torch::Tensor block_y, torch::Tensor block_z, torch::Tensor corner, torch::Tensor IW, 
torch::Tensor book, const int C);
torch::Tensor interpolate_cuda(torch::Tensor val, torch::Tensor point, const int C);

torch::Tensor get_point_backward_cuda(torch::Tensor grad, torch::Tensor nw_val, torch::Tensor ne_val, torch::Tensor sw_val,  
torch::Tensor se_val, const int C);
torch::Tensor interpolate_backward_cuda(torch::Tensor grad, torch::Tensor d_points, torch::Tensor dx_right, torch::Tensor dy_bottom,
torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int N, const int C, const int IW, const int IH);
torch::Tensor interpolate_backward_backward_cuda(torch::Tensor saved_grad_output, torch::Tensor d_points, torch::Tensor x_grad, torch::Tensor y_grad, torch::Tensor dx_right, 
torch::Tensor dy_bottom, torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int IW, const int IH);
torch::Tensor grad_backward_backward_cuda(torch::Tensor d_grad, torch::Tensor x_grad, torch::Tensor y_grad);
std::vector<torch::Tensor> cell_backward_cuda(torch::Tensor grad, torch::Tensor corner, torch::Tensor point,
torch::Tensor IW, torch::Tensor IH, const int C, const int n_block, torch::Tensor book);
// torch::Tensor cell_backward_cuda(torch::Tensor input, torch::Tensor grad, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
// torch::Tensor nw, torch::Tensor ne, torch::Tensor sw, torch::Tensor se,
// const int IW, const int IH, const int C);
// C++
// torch::Tensor normalize(torch::Tensor ix, torch::Tensor iy, const int IW, const int IH) {
torch::Tensor normalize(torch::Tensor coords, torch::Tensor IW, torch::Tensor IH) {
    CHECK_INPUT(coords);
    CHECK_INPUT(IW);
    CHECK_INPUT(IH);
    return normalize_cuda(coords, IW, IH);
}
torch::Tensor normalize_offset(torch::Tensor idx, const int len, torch::Tensor offset) {
    CHECK_INPUT(idx);
    CHECK_INPUT(offset);
    return normalize_offset_cuda(idx, len, offset);
}

torch::Tensor get_corner(torch::Tensor coord) {
    CHECK_INPUT(coord);
    return get_corner_cuda(coord);
}

torch::Tensor get_weight(torch::Tensor coord, torch::Tensor corner) {
    CHECK_INPUT(coord);
    CHECK_INPUT(corner);
    return get_weight_cuda(coord, corner);
}


torch::Tensor get_point(torch::Tensor weight) {
    CHECK_INPUT(weight);
    return get_point_cuda(weight);
}
// torch::Tensor gather(std::vector<torch::Tensor> input, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
// torch::Tensor gather(std::vector<torch::Tensor> input, torch::Tensor corner, torch::Tensor IW, 
// torch::Tensor book, const int C) {
torch::Tensor gather(torch::Tensor block_x, torch::Tensor block_y, torch::Tensor block_z, torch::Tensor corner, torch::Tensor IW, 
torch::Tensor book, const int C) {
    CHECK_INPUT(block_x);
    CHECK_INPUT(block_y);
    CHECK_INPUT(block_z);
    CHECK_INPUT(corner);
    CHECK_INPUT(IW);
    CHECK_INPUT(book);
    return gather_cuda(block_x, block_y, block_z, corner, IW, book, C);
}

torch::Tensor interpolate(torch::Tensor val, torch::Tensor point, const int C) {
    CHECK_INPUT(val);
    CHECK_INPUT(point);
    return interpolate_cuda(val, point, C);
}


torch::Tensor get_point_backward(torch::Tensor grad, torch::Tensor nw_val, torch::Tensor ne_val, torch::Tensor sw_val,  
torch::Tensor se_val, const int C) {
    return get_point_backward_cuda(grad, nw_val, ne_val, sw_val,se_val, C);
}

torch::Tensor interpolate_backward(torch::Tensor grad, torch::Tensor d_points, torch::Tensor dx_right, torch::Tensor dy_bottom,
torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int N, const int C, const int IW, const int IH) {
    return interpolate_backward_cuda(grad, d_points, dx_right, dy_bottom, ix_right, iy_bottom, 
    ix, iy, N,  C,  IW,  IH);
}

torch::Tensor interpolate_backward_backward(torch::Tensor saved_grad_output, torch::Tensor d_points, torch::Tensor x_grad, torch::Tensor y_grad, torch::Tensor dx_right, 
torch::Tensor dy_bottom, torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int IW, const int IH) {
    return interpolate_backward_backward_cuda(saved_grad_output, d_points, x_grad, y_grad, dx_right, dy_bottom, 
    ix_right, iy_bottom, ix, iy, IW, IH);
}

torch::Tensor grad_backward_backward(torch::Tensor d_grad, torch::Tensor x_grad, torch::Tensor y_grad) {
    return grad_backward_backward_cuda(d_grad, x_grad, y_grad);
}
std::vector<torch::Tensor> cell_backward(torch::Tensor grad, torch::Tensor corner, torch::Tensor point,
torch::Tensor IW, torch::Tensor IH, const int C, const int n_block, torch::Tensor book) {
    return cell_backward_cuda(grad, corner, point, IW, IH, C, n_block, book);
}
// torch::Tensor cell_backward(torch::Tensor input, torch::Tensor grad, torch::Tensor ix_right, torch::Tensor ix_left, torch::Tensor iy_bottom, torch::Tensor iy_top,
// torch::Tensor nw, torch::Tensor ne, torch::Tensor sw, torch::Tensor se,
// const int IW, const int IH, const int C) {
//     return cell_backward_cuda(input, grad, ix_right, ix_left, iy_bottom, iy_top, nw, ne, sw, se, IW, IH, C);
// }



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("grid_sample", &grid_sample, "plz");
    m.def("normalize", &normalize, "plz");
    m.def("normalize_offset", &normalize_offset, "plz");
    m.def("get_corner", &get_corner, "plz");
    m.def("get_weight", &get_weight, "plz");
    m.def("get_point", &get_point, "plz");
    m.def("gather", &gather, "plz");
    m.def("interpolate", &interpolate, "plz");
    m.def("get_point_backward", &get_point_backward, "plz");
    m.def("interpolate_backward", &interpolate_backward, "plz");
    m.def("interpolate_backward_backward", &interpolate_backward_backward, "plz");
    m.def("grad_backward_backward", &grad_backward_backward, "plz");
    m.def("cell_backward", &cell_backward, "plz");
}
