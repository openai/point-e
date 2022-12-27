// Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// pyTorchChamferDistance components
// https://github.com/chrdiller/pyTorchChamferDistance
// 
// MIT License
// 
// Copyright (c) 2018 Christian Diller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
// SOFTWARE.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "../utils.h"

#define BLOCK_SIZE 512

namespace kaolin {

template<typename scalar_t>
__global__ void sided_distance_forward_cuda_kernel(
    int b, int n, const scalar_t * xyz,
    int m, const scalar_t * xyz2,
    scalar_t * result, int64_t * result_i) {
  const int batch=512;
  __shared__ scalar_t buf[batch*3];

  for (int i = blockIdx.x; i<b; i += gridDim.x){
    for (int k2 = 0; k2 < m; k2 += batch) {

      int end_k =  min(m, k2 + batch) - k2;

      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j]=xyz2[(i*m+k2)*3+j];
      }

      __syncthreads();

      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz[(i * n + j) * 3 + 0];
        scalar_t y1 = xyz[(i * n + j) * 3 + 1];
        scalar_t z1 = xyz[(i * n + j) * 3 + 2];

        int64_t best_i = 0;
        scalar_t best = 0;
        int end_ka = end_k - (end_k & 3);

        if (end_ka == batch){
          for (int k = 0; k < batch; k += 4) {
            {
            scalar_t x2 = buf[k * 3 + 0] - x1;
            scalar_t y2 = buf[k * 3 + 1] - y1;
            scalar_t z2 = buf[k * 3 + 2]- z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (k == 0 || d < best) {
              best = d;
              best_i = k + k2;
            }
            }

            {
            scalar_t x2 = buf[k * 3 + 3] - x1;
            scalar_t y2 = buf[k * 3 + 4] - y1;
            scalar_t z2 = buf[k * 3 + 5] - z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (d < best){
              best = d;
              best_i = k + k2 + 1;
            }
            }

            {
            scalar_t x2 = buf[k * 3 + 6]- x1;
            scalar_t y2 = buf[k * 3 + 7] - y1;
            scalar_t z2 = buf[k * 3 + 8] - z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (d < best) {
              best = d;
              best_i = k + k2 + 2;
            }
            }

            {
            scalar_t x2 = buf[k * 3 + 9] - x1;
            scalar_t y2 = buf[k * 3 + 10]-y1;
            scalar_t z2 = buf[k*3 + 11] - z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (d < best) {
              best = d;
              best_i = k + k2 + 3;
            }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              scalar_t x2 = buf[k * 3 + 0] - x1;
              scalar_t y2 = buf[k * 3 + 1] - y1;
              scalar_t z2 = buf[k * 3 + 2] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }

            {
              scalar_t x2 = buf[k * 3 + 3] - x1;
              scalar_t y2 = buf[k * 3 + 4] - y1;
              scalar_t z2 = buf[k * 3 + 5] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }

            {
              scalar_t x2 = buf[k * 3 + 6] - x1;
              scalar_t y2 = buf[k * 3 + 7] - y1;
              scalar_t z2 = buf[k * 3 + 8] - z1;
              scalar_t d= x2 * x2 + y2 * y2 + z2 * z2;

              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }

            {
              scalar_t x2 = buf[k * 3 + 9] - x1;
              scalar_t y2 = buf[k * 3 + 10] - y1;
              scalar_t z2 = buf[k * 3 + 11] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2 = buf[k * 3 + 0] - x1;
          scalar_t y2 = buf[k * 3 + 1] - y1;
          scalar_t z2 = buf[k * 3 + 2] - z1;
          scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

          if (k == 0 || d < best) {
            best = d;
            best_i = k+k2;
          }
        }

        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

template<typename scalar_t>
__global__ void sided_distance_backward_cuda_kernel(
    const scalar_t* grad_output,
    const int b,
    const int n,
    const scalar_t * p1,
    const int m,
    const scalar_t * p2,
    const int64_t* idx,
    scalar_t* grad_input1,
    scalar_t* grad_input2) {
  int batch_id = blockIdx.y;
  for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < n; point_id += gridDim.x * blockDim.x) {
    int main_id = point_id + batch_id * n;
    scalar_t x1 = p1[main_id * 3];
    scalar_t y1 = p1[main_id * 3 + 1];
    scalar_t z1 = p1[main_id * 3 + 2];

    int64_t p2_idx = (idx[main_id] + batch_id * m) * 3;

    scalar_t x2 = p2[p2_idx];
    scalar_t y2 = p2[p2_idx + 1];
    scalar_t z2 = p2[p2_idx + 2];

    scalar_t grad = grad_output[main_id];

    grad_input1[main_id * 3] = 2 * (x1 - x2) * grad;
    grad_input1[main_id * 3 + 1] = 2 * (y1 - y2) * grad;
    grad_input1[main_id * 3 + 2] = 2 * (z1 - z2) * grad;

    scalar_t result_x = 2 * (x2 - x1) * grad;
    scalar_t result_y = 2 * (y2 - y1) * grad;
    scalar_t result_z = 2 * (z2 - z1) * grad;

    // compute grad_input2
    atomicAdd(&(grad_input2[p2_idx]), result_x);
    atomicAdd(&(grad_input2[p2_idx + 1]), result_y);
    atomicAdd(&(grad_input2[p2_idx + 2]), result_z);
  }
}

void sided_distance_forward_cuda_impl(
    const at::Tensor p1,
    const at::Tensor p2,
    at::Tensor dist,
    at::Tensor idx) {
  const int batch_size = p1.size(0);
  const int num_p1 = p1.size(1);
  const int num_p2 = p2.size(1);
  DISPATCH_NUM_TYPES(p1.scalar_type(), scalar_t,
                     "sided_distance_forward_cuda", [&] {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(p1));
    auto stream = at::cuda::getCurrentCUDAStream();
    sided_distance_forward_cuda_kernel<scalar_t><<<
      dim3(32, 16, 1), 512, 0, stream>>>(
        batch_size,
        num_p1,
        p1.data_ptr<scalar_t>(),
        num_p2,
        p2.data_ptr<scalar_t>(),
        dist.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
  });
}
void sided_distance_backward_cuda_impl(
    const at::Tensor grad_output,
    const at::Tensor p1,
    const at::Tensor p2,
    const at::Tensor idx,
    const at::Tensor grad_input1,
    const at::Tensor grad_input2) {
  const int batch_size = p1.size(0);
  const int num_p1 = p1.size(1);
  const int num_p2 = p2.size(1);
  const int num_blocks = (max(num_p1, num_p2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  DISPATCH_NUM_TYPES(p1.scalar_type(), scalar_t,
                     "sided_distance_backward_cuda", [&] {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(p1));
    auto stream = at::cuda::getCurrentCUDAStream();
    sided_distance_backward_cuda_kernel<scalar_t><<<
      dim3(num_blocks, batch_size, 1), BLOCK_SIZE, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        batch_size,
        num_p1,
        p1.data_ptr<scalar_t>(),
        num_p2,
        p2.data_ptr<scalar_t>(),
        idx.data_ptr<int64_t>(),
        grad_input1.data_ptr<scalar_t>(),
        grad_input2.data_ptr<scalar_t>());
    AT_CUDA_CHECK(cudaGetLastError());
  });
}
#undef BLOCK_SIZE
}  // namespace kaolin

