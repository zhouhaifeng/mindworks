// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/util/fft_base.hpp>

#include "openvino/core/axis_vector.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const util::FFTBase* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using DimType = typename T::value_type;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3));

    const auto& input_shape = input_shapes[0];
    const auto& axes_shape = input_shapes[1];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    auto axes = get_input_const_data_as<TRShape, int64_t>(op, 1, ta);

    if (input_shape.rank().is_static()) {
        const auto input_rank = input_shape.size();
        NODE_VALIDATION_CHECK(op,
                              input_rank >= 2,
                              "The input rank must be greater or equal to 2. Got input rank: ",
                              input_rank);

        NODE_VALIDATION_CHECK(op,
                              input_shape[input_rank - 1].compatible(2),
                              "The last dimension of input data must be 2. Got: ",
                              input_shape[input_rank - 1]);

        if (axes_shape.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  input_rank >= static_cast<size_t>(axes_shape[0].get_length() + 1),
                                  "The input rank must be greater than number of FFT op axes. Got "
                                  "input rank: ",
                                  input_rank,
                                  ", number of axes: ",
                                  axes_shape[0].get_length());
        }

        // FFT operation supports for negative axes to transform. More precisely, according to
        // the FFT operation specification, axes should be integers from -(r - 1) to (r - 2)
        // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis
        // 'r - 1 + a'. The reason is the following: real input tensor of the shape
        // [n_0, ..., n_{r - 1}, 2] is interpreted as a complex tensor with the shape
        // [n_0, ..., n_{r - 1}].
        if (axes_shape.rank().is_static() && axes) {
            const auto axis_min_value = -static_cast<int64_t>(input_rank);
            const auto axis_max_value = static_cast<int64_t>(input_rank) - 1;
            ov::AxisSet axes_set;
            for (int64_t& axis : *axes) {
                NODE_VALIDATION_CHECK(op,
                                      axis_min_value < axis && axis < axis_max_value,
                                      "FFT op axis ",
                                      axis,
                                      " must be in the input rank range (",
                                      axis_min_value,
                                      ", ",
                                      axis_max_value,
                                      ").");
                if (axis < 0) {
                    axis += input_rank - 1;
                }
                axes_set.insert(static_cast<size_t>(axis));
            }

            NODE_VALIDATION_CHECK(op, axes->size() == axes_set.size(), "FFT op axes must be unique.");
        }
    }

    NODE_VALIDATION_CHECK(op, axes_shape.rank().compatible(1), "FFT op axes input must be 1D tensor.");

    if (input_shapes.size() == 3) {
        const auto& signal_size_shape = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              signal_size_shape.rank().compatible(1),
                              "FFT op signal size input must be 1D tensor. Got signal: ",
                              signal_size_shape);

        if (axes_shape.is_static() && signal_size_shape.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  axes_shape[0].compatible(signal_size_shape[0]),
                                  "Sizes of inputs 'axes' and 'signal_size' must be equal. Got "
                                  "size of 'axes': ",
                                  axes_shape[0],
                                  "size of 'signal_size': ",
                                  signal_size_shape[0]);
        }
    }

    output_shape = input_shape;
    if (input_shape.rank().is_static() && axes_shape.rank().is_static() && input_shapes.size() == 3 && axes) {
        const auto& signal_size_shape = input_shapes[2];
        auto signal_size = get_input_const_data_as<TRShape, int64_t>(op, 2, ta);

        if (signal_size_shape.rank().is_static() && signal_size) {
            size_t num_of_axes = axes->size();
            for (size_t i = 0; i < num_of_axes; ++i) {
                if ((*signal_size)[i] == -1) {
                    continue;
                }
                output_shape[(*axes)[i]] = DimType((*signal_size)[i]);
            }
        } else if (signal_size_shape.rank().is_static()) {
            for (int64_t& axis : *axes) {
                output_shape[axis] = ov::Dimension::dynamic();
            }
        }
    } else if (input_shape.rank().is_static() && (axes_shape.rank().is_dynamic() || !axes)) {
        const auto input_rank = input_shape.size();
        for (size_t i = 0; i < input_rank - 1; ++i) {
            output_shape[i] = ov::Dimension::dynamic();
        }
    }
    return output_shapes;
}
}  // namespace op
}  // namespace ov
