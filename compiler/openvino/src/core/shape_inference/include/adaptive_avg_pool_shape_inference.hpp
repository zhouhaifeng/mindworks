// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/adaptive_avg_pool.hpp"
#include "pooling_shape_inference_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v8 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const AdaptiveAvgPool* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_acessor = make_tensor_accessor()) {
    return {pooling::out_shape_infer(op, input_shapes, tensor_acessor)};
}
}  // namespace v8
}  // namespace op
}  // namespace ov
