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

#include "../check.h"

namespace point_e {

#ifdef WITH_CUDA
void sided_distance_forward_cuda_impl(
    const at::Tensor p1,
    const at::Tensor p2,
    at::Tensor dist,
    at::Tensor idx);

void sided_distance_backward_cuda_impl(
    at::Tensor grad_output,
    at::Tensor p1,
    at::Tensor p2,
    at::Tensor idx,
    at::Tensor grad_input1,
    at::Tensor grad_input2);

#endif  // WITH_CUDA


std::vector<at::Tensor> sided_distance_forward_cuda(
    const at::Tensor p1,
    const at::Tensor p2) {
  at::TensorArg p1_arg{p1, "p1", 1}, p2_arg{p2, "p2", 2};
  at::checkSameGPU(__func__, p1_arg, p2_arg);
  at::checkAllContiguous(__func__, {p1_arg, p2_arg});
  at::checkSameType(__func__, p1_arg, p2_arg);

  const int batch_size = p1.size(0);
  const int num_p1 = p1.size(1);
  const int num_p2 = p2.size(1);

  at::checkSize(__func__, p1_arg, {batch_size, num_p1, 3});
  at::checkSize(__func__, p2_arg, {batch_size, num_p2, 3});

  auto dist = at::zeros({batch_size, num_p1}, p1.options());
  auto idx = at::zeros({batch_size, num_p1}, p1.options().dtype(at::kLong));

#ifdef WITH_CUDA
  sided_distance_forward_cuda_impl(p1, p2, dist, idx);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
  return {dist, idx};
}

std::vector<at::Tensor> sided_distance_backward_cuda(
    at::Tensor grad_output,
    at::Tensor p1,
    at::Tensor p2,
    at::Tensor idx) {
  at::TensorArg grad_output_arg{grad_output, "grad_output", 1};
  at::TensorArg p1_arg{p1, "p1", 2};
  at::TensorArg p2_arg{p2, "p2", 3};
  at::TensorArg idx_arg{idx, "idx", 4};

  at::checkAllSameGPU(__func__, {grad_output_arg, p1_arg, p2_arg, idx_arg});
  at::checkAllContiguous(__func__, {grad_output_arg, p1_arg, p2_arg, idx_arg});

  const int batch_size = p1.size(0);
  const int num_p1 = p1.size(1);
  const int num_p2 = p2.size(1);

  at::checkSize(__func__, idx_arg, {batch_size, num_p1});
  at::checkSize(__func__, p1_arg, {batch_size, num_p1, 3});
  at::checkSize(__func__, p2_arg, {batch_size, num_p2, 3});
  at::checkSameSize(__func__, idx_arg, grad_output_arg);

  auto grad_input1 = at::zeros_like(p1);
  auto grad_input2 = at::zeros_like(p2);

#ifdef WITH_CUDA
  sided_distance_backward_cuda_impl(grad_output, p1, p2, idx, grad_input1, grad_input2);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
  return {grad_input1, grad_input2};
}

}  // namespace kaolin