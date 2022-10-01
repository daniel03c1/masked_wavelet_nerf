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

void global_to_local_cuda(const torch::Tensor& points_tensor, const torch::Tensor& domain_mins_tensor, const torch::Tensor& domain_maxs_tensor,
    const torch::Tensor& batch_size_per_network_tensor, int64_t kernel_num_blocks, int64_t kernel_num_threads);

void global_to_local(const torch::Tensor& points_tensor, const torch::Tensor& domain_mins_tensor, const torch::Tensor& domain_maxs_tensor,
    const torch::Tensor& batch_size_per_network_tensor, int64_t kernel_num_blocks, int64_t kernel_num_threads) {
        global_to_local_cuda(points_tensor, domain_mins_tensor, domain_maxs_tensor,
    batch_size_per_network_tensor, kernel_num_blocks, kernel_num_threads); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("global_to_local", &global_to_local, "");
}
