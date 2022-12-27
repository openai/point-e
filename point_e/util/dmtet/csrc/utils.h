

// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_UTILS_H_
#define KAOLIN_UTILS_H_

#include <ATen/ATen.h>
#include <typeinfo>
#include <cuda.h>

#define PRIVATE_CASE_TYPE(ENUM_TYPE, TYPE, TYPE_NAME, ...) \
  case ENUM_TYPE: { \
    using TYPE_NAME = TYPE; \
    return __VA_ARGS__(); \
  }

#define PRIVATE_CASE_INOUT_TYPES(CONST_IN_TYPE, CONST_OUT_TYPE, ENUM_IN_TYPE, ENUM_OUT_TYPE, \
                                 IN_TYPE, OUT_TYPE, IN_TYPE_NAME, OUT_TYPE_NAME, ...) \
  if (CONST_IN_TYPE == ENUM_IN_TYPE && CONST_OUT_TYPE == ENUM_OUT_TYPE) { \
    using IN_TYPE_NAME = IN_TYPE; \
    using OUT_TYPE_NAME = OUT_TYPE; \
    return __VA_ARGS__(); \
  } else \

#define PRIVATE_CASE_INOUT_DEDUCED_TYPES(ENUM_TYPE, IN_TYPE, OUT_TYPE, \
                                         IN_TYPE_NAME, OUT_TYPE_NAME, ...) \
  case ENUM_TYPE: { \
    using IN_TYPE_NAME = IN_TYPE; \
    using OUT_TYPE_NAME = OUT_TYPE; \
    return __VA_ARGS__(); \
  }

#define PRIVATE_CASE_INT(CONST_INT, VAR_NAME, ...) \
  case CONST_INT: { \
    const int VAR_NAME = CONST_INT; \
    return __VA_ARGS__(); \
  }

#define DISPATCH_NUM_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Int, int, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Float, float, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Double, double, TYPE_NAME, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()


#define DISPATCH_INTEGER_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Int, int, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, TYPE_NAME, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()

#define DISPATCH_FLOAT_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Float, float, TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE(at::ScalarType::Double, double, TYPE_NAME, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()

#endif  // KAOLIN_UTILS_H_