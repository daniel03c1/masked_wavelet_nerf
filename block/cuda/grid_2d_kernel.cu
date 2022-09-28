#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <iostream>
#include <math_constants.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <memory.h>

#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define CUDA_MAX_THREADS at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

using namespace std;

// 이것도 일단은 if문.. TODO!!!!!! 
template <typename scalar_t>
__global__ void normalize_kernel(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> idx, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret, 
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> IW, 
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> IH, 
const int ele_num) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int axis = blockIdx.y; // 0 1 2 . 어느 블록에 대한건지

  if (c < ele_num) {
    if (axis == 0) {
      ret[axis][0][c] = ((idx[c][2]+1)/2)*IW[2];  
      ret[axis][1][c] = ((idx[c][1]+1)/2)*IH[1];  
    }
    else if (axis == 1) {
      ret[axis][0][c] = ((idx[c][2]+1)/2)*IW[2];  
      ret[axis][1][c] = ((idx[c][0]+1)/2)*IH[0];  
    }
    else {
      ret[axis][0][c] = ((idx[c][0]+1)/2)*IW[0];  
      ret[axis][1][c] = ((idx[c][1]+1)/2)*IH[1];  
    }
  }
}

template <typename scalar_t>
__global__ void normalize_offset_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> idx, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret, const int len, 
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> offset) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  
  const int n_points = idx.size(2);
  const int n_cells = idx.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  
  if (c < n_points && n < n_cells) { 
    ret[n][0][c] = ((idx[n][0][c]+1)/2)*len + offset[n];
  }
}

template <typename scalar_t>
__global__ void step_bilinear_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coord, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> corner,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret, const int ele_num) { 
  int c = blockIdx.x * blockDim.x + threadIdx.x;  
  int b = blockIdx.y;

  if (c < ele_num) {
    scalar_t weight_x = corner[b][1][c] - coord[b][0][c];
    scalar_t weight_y = corner[b][3][c] - coord[b][1][c];
    
    ret[b][0][c] = weight_x;
    ret[b][1][c] = 1 - weight_x;
    ret[b][2][c] = weight_y;
    ret[b][3][c] = 1 - weight_y;
  }
}

template <typename scalar_t>
__global__ void step_cosine_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  
  const int n_points = ix.size(2);
  const int n_cells = ix.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  

  if (c < n_points && n < n_cells) {
    scalar_t tmp_ix = ix_right[n][0][c] - ix[n][0][c];
    scalar_t tmp_iy = iy_bottom[n][0][c] - iy[n][0][c];
    ret[0][n][0][c] = 0.5*(1-cos(CUDART_PI_F*tmp_ix)); 
    ret[1][n][0][c] = 1 - ret[0][n][0][c];
    ret[2][n][0][c] = 0.5*(1-cos(CUDART_PI_F*tmp_iy)); 
    ret[3][n][0][c] = 1 - ret[2][n][0][c];
  }
}

template <typename scalar_t>
__global__ void step_smooth_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix, 
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int n = blockIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (c < ix.size(2) * ix.size(0)) {
    scalar_t tmp_ix = ix_right[n][0][c] - ix[n][0][c];
    scalar_t tmp_iy = iy_bottom[n][0][c] - iy[n][0][c];
    ret[0][n][0][c] = pow(tmp_ix,2) * (3-2*tmp_ix); 
    ret[1][n][0][c] = 1 - ret[0][n][0][c];
    ret[2][n][0][c] = pow(tmp_iy,2) * (3-2*tmp_iy); 
    ret[3][n][0][c] = 1 - ret[2][n][0][c];
  }
}

template <typename scalar_t>
__global__ void get_corner_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> coord,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret, const int ele_num) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;  
  int b = blockIdx.y; // 0, 1, 2
  if (c < ele_num) {
    scalar_t corner_x = floorf(coord[b][0][c]);
    scalar_t corner_y = floorf(coord[b][1][c]);
    ret[b][0][c] = corner_x;
    ret[b][1][c] = corner_x + 1; 
    ret[b][2][c] = corner_y;
    ret[b][3][c] = corner_y + 1;
  }
}


template <typename scalar_t>
__global__ void compute_point_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weight,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ret, const int ele_num) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;  
  int b = blockIdx.y;

  if (c < ele_num) {
    ret[b][0][c] = weight[b][0][c]*weight[b][2][c];
    ret[b][1][c] = weight[b][1][c]*weight[b][2][c];
    ret[b][2][c] = weight[b][0][c]*weight[b][3][c];
    ret[b][3][c] = weight[b][1][c]*weight[b][3][c];
  }
}


// TODO if!!!!
template <typename scalar_t> 
// __global__ void gather_kernel(torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input, 
// __global__ void gather_kernel(std::vector<torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>> input, 
// __global__ void gather_kernel(scalar_t* input, 
// __global__ void gather_kernel(std::vector<torch::Tensor> input, 
__global__ void gather_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> block_x,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> block_y,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> block_z,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> corner,
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> IW, 
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> book, 
const int C, const int ele_num,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  int coef = blockIdx.z;

  if (c < ele_num && coef < C) {  
    const int iw_b = IW[b];

    long idx_nw = corner[b][2][c] * iw_b + corner[b][0][c];
    long idx_ne = corner[b][2][c] * iw_b + corner[b][1][c];
    long idx_sw = corner[b][3][c] * iw_b + corner[b][0][c];
    long idx_se = corner[b][3][c] * iw_b + corner[b][1][c];

    long cur_block = book[c];


    if (b == 0) {
      ret[b][1][coef][c] =  block_x[cur_block][0][coef][idx_ne];
      ret[b][2][coef][c] =  block_x[cur_block][0][coef][idx_sw];
      ret[b][3][coef][c] =  block_x[cur_block][0][coef][idx_se];
      ret[b][0][coef][c]  = block_x[cur_block][0][coef][idx_nw];
    }
    else if (b == 1) {
      ret[b][1][coef][c] =  block_y[cur_block][0][coef][idx_ne];
      ret[b][2][coef][c] =  block_y[cur_block][0][coef][idx_sw];
      ret[b][3][coef][c] =  block_y[cur_block][0][coef][idx_se];
      ret[b][0][coef][c]  = block_y[cur_block][0][coef][idx_nw];
    }
    else {
      ret[b][1][coef][c] =  block_z[cur_block][0][coef][idx_ne];
      ret[b][2][coef][c] =  block_z[cur_block][0][coef][idx_sw];
      ret[b][3][coef][c] =  block_z[cur_block][0][coef][idx_se];
      ret[b][0][coef][c]  = block_z[cur_block][0][coef][idx_nw];
    }
  }
}

template <typename scalar_t>
__global__ void interpolate_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> val,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> point,
const int C,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intpl_val, const int ele_num) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int axis = blockIdx.y;
  int coef = blockIdx.z;

  if (c < ele_num && coef < C) { 
    intpl_val[axis][coef][c] = val[axis][0][coef][c] * point[axis][0][c] + val[axis][1][coef][c] * point[axis][1][c] + 
      val[axis][2][coef][c] * point[axis][2][c] + val[axis][3][coef][c] * point[axis][3][c];

  }

}


__device__ float atomicAddDouble(float* address, float val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


// l r t b
template <typename scalar_t> 
__global__ void cell_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> corner,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> point,
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> IW,
const int C,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret_x,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret_y,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ret_z,
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> book, const int ele_num) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int axis = blockIdx.y;
  int coef = blockIdx.z;

  if (c < ele_num && coef < C) { 
    int iw_b = IW[axis];
    long idx_nw = corner[axis][2][c] * iw_b + corner[axis][0][c]; 
    long idx_ne = corner[axis][2][c] * iw_b + corner[axis][1][c];
    long idx_sw = corner[axis][3][c] * iw_b + corner[axis][0][c];
    long idx_se = corner[axis][3][c] * iw_b + corner[axis][1][c];

    long cur_block = book[c];

    if (axis == 0) {
      atomicAdd(&ret_x[cur_block][0][coef][idx_nw] ,grad[axis][coef][c] * point[axis][0][c]);
      atomicAdd(&ret_x[cur_block][0][coef][idx_ne] ,grad[axis][coef][c] * point[axis][1][c]);
      atomicAdd(&ret_x[cur_block][0][coef][idx_sw] ,grad[axis][coef][c] * point[axis][2][c]);
      atomicAdd(&ret_x[cur_block][0][coef][idx_se] ,grad[axis][coef][c] * point[axis][3][c]);
    }
    else if (axis == 1) {
      atomicAdd(&ret_y[cur_block][0][coef][idx_nw] ,grad[axis][coef][c] * point[axis][0][c]);
      atomicAdd(&ret_y[cur_block][0][coef][idx_ne] ,grad[axis][coef][c] * point[axis][1][c]);
      atomicAdd(&ret_y[cur_block][0][coef][idx_sw] ,grad[axis][coef][c] * point[axis][2][c]);
      atomicAdd(&ret_y[cur_block][0][coef][idx_se] ,grad[axis][coef][c] * point[axis][3][c]);
    }
    else {
      atomicAdd(&ret_z[cur_block][0][coef][idx_nw] ,grad[axis][coef][c] * point[axis][0][c]);
      atomicAdd(&ret_z[cur_block][0][coef][idx_ne] ,grad[axis][coef][c] * point[axis][1][c]);
      atomicAdd(&ret_z[cur_block][0][coef][idx_sw] ,grad[axis][coef][c] * point[axis][2][c]);
      atomicAdd(&ret_z[cur_block][0][coef][idx_se] ,grad[axis][coef][c] * point[axis][3][c]);
    }
    

  }
}




template <typename scalar_t> 
__global__ void migrate_kernel(int* idx, float* val, const int num_ele, 
torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ret,
const int n_cells) {
    int tmp = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp < num_ele) {
      ret[idx[tmp]] = val[tmp];
    }

}


template <typename scalar_t>
__global__ void get_point_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> nw_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> ne_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> sw_val,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> se_val,
const int C,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> ret) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = nw_val.size(3);
  const int n_cells = nw_val.size(0);
  int n = tmp / n_points;
  int c = tmp % n_points;  

  if (c < n_points && n < n_cells) {
#pragma unroll
    for (int i = 0; i<C; i++) {
      ret[0][n][i][0][c] = grad[n][i][0][c] * nw_val[n][i][0][c];
      ret[1][n][i][0][c] = grad[n][i][0][c] * ne_val[n][i][0][c];
      ret[2][n][i][0][c] = grad[n][i][0][c] * sw_val[n][i][0][c];
      ret[3][n][i][0][c] = grad[n][i][0][c] * se_val[n][i][0][c];
    }
  }

}

template <typename scalar_t>
__global__ void interpolate_backward_kernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> d_points,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
const int IW, const int IH,
torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> d_grad) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = d_points.size(4);
  const int n_cells = d_points.size(1); 
  const int C = d_points.size(2); 
  int n = tmp / n_points;
  int c = tmp % n_points;  
  


  if (c < n_points && n < n_cells) {   
#pragma unroll
    for (int i = 0; i<C; i++) {
      d_grad[0][n][i][0][c][0] = (-IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][n][i][0][c] - d_points[1][n][i][0][c]) + \
            (1 - dy_bottom[n][0][c]) * (d_points[2][n][i][0][c] - d_points[3][n][i][0][c]))) * grad[n][i][0][c];
      d_grad[1][n][i][0][c][0] = (-IH * 0.5 * (dx_right[n][0][c] *(d_points[0][n][i][0][c] - d_points[2][n][i][0][c]) + \
            (1 - dx_right[n][0][c]) * (d_points[1][n][i][0][c] - d_points[3][n][i][0][c]))) * grad[n][i][0][c]; 
    }
  }

}


template <typename scalar_t>
__global__ void interpolate_backward_backward_kernel(
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> saved_grad_out,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> x_grad,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> y_grad,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> d_points,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix_right,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy_bottom,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> ix,
torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> iy,
const int IW, const int IH,
torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> dd_grad) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = d_points.size(4);
  const int n_cells = d_points.size(1); 
  const int C = d_points.size(2); 
  int n = tmp / n_points;
  int c = tmp % n_points;  
  

  if (c < n_points && n < n_cells) {   
#pragma unroll
    for (int i = 0; i<C; i++) {
      
      dd_grad[0][n][i][0][c][0] = CUDART_PI_F * CUDART_PI_F * 0.5 * cos((ix_right[n][0][c] - ix[n][0][c])*CUDART_PI_F) * IW * 0.5 * (dy_bottom[n][0][c] *(d_points[0][n][i][0][c] - d_points[1][n][i][0][c]) + \
            (1 - dy_bottom[n][0][c]) * (d_points[2][n][i][0][c] - d_points[3][n][i][0][c])) * IW * 0.5 * x_grad[n][i][0][c][0] * saved_grad_out[n][i][0][c] ;

      dd_grad[1][n][i][0][c][0] = CUDART_PI_F * CUDART_PI_F * 0.5 * cos((iy_bottom[n][0][c] - iy[n][0][c])*CUDART_PI_F) * IH * 0.5 * (dx_right[n][0][c] *(d_points[0][n][i][0][c] - d_points[2][n][i][0][c]) + \
            (1 - dx_right[n][0][c]) * (d_points[1][n][i][0][c] - d_points[3][n][i][0][c])) * IH * 0.5 * y_grad[n][i][0][c][0] * saved_grad_out[n][i][0][c] ;
      
    }
  }

}
template <typename scalar_t>
__global__ void grad_backward_backward_kernel(
torch::PackedTensorAccessor32<scalar_t, 6, torch::RestrictPtrTraits> d_grad,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> x_grad,
torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> y_grad,
torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dd_grad) {
  int tmp = blockIdx.x * blockDim.x + threadIdx.x;  

  const int n_points = x_grad.size(3);
  const int n_cells = x_grad.size(0); 
  const int C = x_grad.size(1); 
  int n = tmp / n_points;
  int c = tmp % n_points;  
  

  if (c < n_points && n < n_cells) {   
#pragma unroll
    for (int i = 0; i<C; i++) {
      dd_grad[n][i][0][c] = d_grad[0][n][i][0][c][0] * x_grad[n][i][0][c][0] + d_grad[1][n][i][0][c][0] * y_grad[n][i][0][c][0];
    }
  }


}



torch::Tensor normalize_cuda(torch::Tensor coords, torch::Tensor IW, torch::Tensor IH) {
  const int ele_num = coords.size(0);
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks(CUDA_N_BLOCKS_NEEDED(ele_num, threads), 3); 
  torch::Tensor ret = torch::empty({3, 2, ele_num}, coords.options());
    AT_DISPATCH_FLOATING_TYPES(coords.type(), "normalize_kernel", ([&] {
      normalize_kernel<scalar_t><<<blocks, threads>>>(coords.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
      IW.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(), 
      IH.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(), ele_num);
    }));
  cudaDeviceSynchronize();
  return ret;
}

torch::Tensor normalize_offset_cuda(torch::Tensor input, int len, torch::Tensor offset) {
  int ele_num = input.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (input.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads)); 
  torch::Tensor ret = torch::empty_like(input, input.options());
    AT_DISPATCH_FLOATING_TYPES(input.type(), "normalize_offset_kernel", ([&] {
      normalize_offset_kernel<scalar_t><<<blocks, threads>>>(input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
      ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), len,
      offset.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
    }));
  cudaDeviceSynchronize();
  return ret;
}

torch::Tensor get_corner_cuda(torch::Tensor coord) {
  const int ele_num = coord.size(2);
  const int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  const dim3 blocks(CUDA_N_BLOCKS_NEEDED(ele_num, threads), 3);
  torch::Tensor ret = torch::zeros({3, 4, ele_num}, coord.options());
  AT_DISPATCH_FLOATING_TYPES(coord.type(), "get_corner_kernel", ([&] {
    get_corner_kernel<scalar_t><<<blocks, threads>>>(coord.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), ele_num);
  }));
  cudaDeviceSynchronize();
 
  return ret;
}



torch::Tensor get_weight_cuda(torch::Tensor coord, torch::Tensor corner) {
  int ele_num = coord.size(2);
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks(CUDA_N_BLOCKS_NEEDED(ele_num, threads), 3);
  torch::Tensor ret = torch::empty({3, 4, ele_num}, coord.options());
  AT_DISPATCH_FLOATING_TYPES(coord.type(), "step_bilinear", ([&] {
    step_bilinear_kernel<scalar_t><<<blocks, threads>>>(coord.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    corner.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), ele_num);
  }));
  cudaDeviceSynchronize();
  return ret;
}


torch::Tensor get_point_cuda(torch::Tensor weight) {
  int ele_num = weight.size(2);
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks(CUDA_N_BLOCKS_NEEDED(ele_num, threads), 3);

  torch::Tensor ret = torch::empty({3, 4, ele_num}, weight.options());
  AT_DISPATCH_FLOATING_TYPES(weight.type(), "compute_point_kernel", ([&] {
    compute_point_kernel<scalar_t><<<blocks, threads>>>(weight.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), ele_num);
  }));
  cudaDeviceSynchronize();
  return ret;
}

// torch::Tensor gather_cuda(std::vector<torch::Tensor> input, torch::Tensor corner, torch::Tensor IW, 
torch::Tensor gather_cuda(torch::Tensor block_x, torch::Tensor block_y, torch::Tensor block_z, torch::Tensor corner, torch::Tensor IW, 
torch::Tensor book, const int C) {
  int ele_num = corner.size(2);
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks(CUDA_N_BLOCKS_NEEDED(ele_num, threads), 3, C);
  torch::Tensor ret = torch::empty({3,4, C, ele_num}, corner.options());

  // cout << *book[0].data<long>() <<" " << *book[1234].data<long>() << endl;
  // cout << book[0].item<long>() <<" " << book[1234].item<long>() << endl;
  // cout << book.type() << endl;

  AT_DISPATCH_FLOATING_TYPES(corner.type(), "gather_kernel", ([&] {
    gather_kernel<scalar_t><<<blocks, threads>>>(
    block_x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    block_y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    block_z.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    corner.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    IW.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
    book.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
    C, ele_num,
    ret.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return ret;
} 

torch::Tensor interpolate_cuda(torch::Tensor val, torch::Tensor point, const int C) {
  int ele_num = val.size(3); // E
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks(CUDA_N_BLOCKS_NEEDED(ele_num, threads),3, C); 

  torch::Tensor ret = torch::zeros({3, C, ele_num}, val.options());


  AT_DISPATCH_FLOATING_TYPES(val.type(), "interpolate_kernel", ([&] {
    interpolate_kernel<scalar_t><<<blocks, threads>>>(val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    point.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    C,
    ret.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), ele_num);
  }));
  cudaDeviceSynchronize();
  return ret;
}

// APP1
std::vector<torch::Tensor> cell_backward_cuda(torch::Tensor grad, torch::Tensor corner, torch::Tensor point,
torch::Tensor IW, torch::Tensor IH, const int C, const int n_block, torch::Tensor book) {
  int ele_num = corner.size(2);
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks(CUDA_N_BLOCKS_NEEDED(ele_num, threads), 3, C);
  torch::Tensor ret_x = torch::zeros({n_block, 1, C, int(IH[0].item<float>()) * int(IW[0].item<float>())}, point.options());
  torch::Tensor ret_y = torch::zeros({n_block, 1, C, int(IH[1].item<float>()) * int(IW[1].item<float>())}, point.options());
  torch::Tensor ret_z = torch::zeros({n_block, 1, C, int(IH[2].item<float>()) * int(IW[2].item<float>())}, point.options());


  AT_DISPATCH_FLOATING_TYPES(corner.type(), "cell_backward_kernel", ([&] {
    cell_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    corner.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    point.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    IW.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
    C,
    ret_x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ret_y.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ret_z.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    book.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(), ele_num);
  }));
  cudaDeviceSynchronize();
  // // ret = ret.view({n_cells, C, IW, IH });
  return {ret_x, ret_y, ret_z};
} 




torch::Tensor get_point_backward_cuda(torch::Tensor grad, torch::Tensor nw_val, torch::Tensor ne_val, torch::Tensor sw_val,  
torch::Tensor se_val, const int C) {
  int ele_num = grad.numel() / C;   
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (grad.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));

  torch::Tensor d_points = torch::empty({4, grad.size(0), grad.size(1),grad.size(2),grad.size(3)}, torch::requires_grad().device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "get_point_backward_kernel", ([&] {
    get_point_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    nw_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    ne_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    sw_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    se_val.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), 
    C,
    d_points.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return d_points;
}

torch::Tensor interpolate_backward_cuda(torch::Tensor grad, torch::Tensor d_points, torch::Tensor dx_right, torch::Tensor dy_bottom,
torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int N, const int C, const int IW, const int IH) {

  int ele_num = grad.numel() / C;   
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (grad.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  torch::Tensor d_grad = torch::zeros({4, grad.size(0), grad.size(1),grad.size(2),grad.size(3), 1}, torch::requires_grad().device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "interpolate_backward_kernel", ([&] {
    interpolate_backward_kernel<scalar_t><<<blocks, threads>>>(grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    d_points.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    dx_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    dy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    IW, IH,
    d_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return d_grad;
}



torch::Tensor interpolate_backward_backward_cuda(torch::Tensor saved_grad_out, torch::Tensor d_points, torch::Tensor x_grad, torch::Tensor y_grad, torch::Tensor dx_right, 
torch::Tensor dy_bottom, torch::Tensor ix_right, torch::Tensor iy_bottom, torch::Tensor ix, torch::Tensor iy,
const int IW, const int IH) {
  int ele_num = saved_grad_out.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (saved_grad_out.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  torch::Tensor dd_grad = torch::zeros({2, saved_grad_out.size(0), saved_grad_out.size(1),saved_grad_out.size(2),saved_grad_out.size(3), 1}, saved_grad_out.options());
  AT_DISPATCH_FLOATING_TYPES(saved_grad_out.type(), "interpolate_backward_backward_kernel", ([&] {
    interpolate_backward_backward_kernel<scalar_t><<<blocks, threads>>>(saved_grad_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
    x_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    y_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    d_points.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    dx_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    dy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    ix_right.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    iy_bottom.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(), 
    ix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    iy.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    IW, IH,
    dd_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return dd_grad;
}

torch::Tensor grad_backward_backward_cuda(torch::Tensor d_grad, torch::Tensor x_grad, torch::Tensor y_grad) {
  int ele_num = x_grad.numel();
  int threads = std::min<int>(CUDA_MAX_THREADS, ele_num); 
  dim3 blocks = (x_grad.size(0), CUDA_N_BLOCKS_NEEDED(ele_num, threads));
  torch::Tensor dd_grad = torch::zeros({x_grad.size(0), x_grad.size(1),x_grad.size(2),x_grad.size(3)}, x_grad.options());
  AT_DISPATCH_FLOATING_TYPES(x_grad.type(), "grad_backward_backward_kernel", ([&] {
    grad_backward_backward_kernel<scalar_t><<<blocks, threads>>>(d_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
    x_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    y_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
    dd_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));
  cudaDeviceSynchronize();
  return dd_grad;
}

