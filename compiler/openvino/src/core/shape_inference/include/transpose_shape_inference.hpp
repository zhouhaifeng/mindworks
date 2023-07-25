// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

/**
 * \brief Calculate transpose output shape.
 *
 * \tparam T           Type of shape
 *
 * \param op           Transpose operator pointer.
 * \param input_shape  Transpose input shape.
 * \param axes_order   Transpose axes order (modified if empty).
 *
 * \return Output shape
 */
template <class T, class TRShape = result_shape_t<T>>
TRShape calc_output_shape(const Transpose* const op, const T& input_shape, std::vector<int64_t>& axes_order) {
    const auto output_rank = input_shape.size();

    if (axes_order.empty()) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        generate_transpose_default_order(axes_order, output_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
    } else {
        OPENVINO_SUPPRESS_DEPRECATED_START
        NODE_VALIDATION_CHECK(op,
                              is_valid_axes_order(axes_order, output_rank),
                              "Permutation ",
                              AxisVector(axes_order.begin(), axes_order.end()),
                              " is not valid for input shape ",
                              input_shape);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    TRShape output_shape;
    for (auto&& axis : axes_order) {
        output_shape.push_back(input_shape[axis]);
    }

    return output_shape;
}

/**
 * \brief Do transpose shape inference on input and output shapes.
 *
 * \tparam TShape          Type of input shapes.
 * \tparam TRShape         Type of return shapes.
 *
 * \param op               Transpose operator pointer.
 * \param input_shapes     Input shapes of transpose.
 * \param tensor_accessor  Accessor to constant data.
 */
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Transpose* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    const auto& input_shape = input_shapes[Transpose::ARG];

    const auto axes = get_input_const_data_as<TShape, int64_t>(op, Transpose::ORDER, tensor_accessor);

    auto output_shapes = std::vector<TRShape>();
    if (axes && input_shape.rank().is_static()) {
        output_shapes.push_back(calc_output_shape(op, input_shape, *axes));
    } else if (axes) {
        output_shapes.push_back(ov::PartialShape::dynamic(axes->size()));
    } else {
        output_shapes.push_back(ov::PartialShape::dynamic(input_shape.rank()));
    }
    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
