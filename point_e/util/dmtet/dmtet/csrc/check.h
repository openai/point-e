// Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_CHECK_H_
#define KAOLIN_CHECK_H_

#include <ATen/native/TypeProperties.h>
#include <ATen/TensorGeometry.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a cpu tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, #x " is not half")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be byte")
#define CHECK_DOUBLE(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Double, #x " must be double")
#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, #x " must be bool")
#define CHECK_BYTE(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Byte, #x " must be byte")
#define CHECK_SHORT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Short, #x " must be short")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be long")

#define CHECK_DIMS(x, d) TORCH_CHECK(x.dim() == d, #x " must have " #d " dims")
#define CHECK_SIZE(x, d, s) \
  TORCH_CHECK(x.size(d) == s, #x " must have dim " #d " of size " #s)
#define CHECK_SIZES(x, ...)                                     \
  TORCH_CHECK(x.sizes() == std::vector<int64_t>({__VA_ARGS__}), \
      #x " must of size {" #__VA_ARGS__ "}")

#define KAOLIN_NO_CUDA_ERROR(func_name)                       \
  AT_ERROR("In ", func_name, ": Kaolin built without CUDA, "  \
           "cannot run with GPU tensors")

#endif  // KAOLIN_CHECK_H_