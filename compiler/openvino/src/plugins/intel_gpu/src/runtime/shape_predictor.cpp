// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/shape_predictor.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace cldnn {

static ov::Shape operator*(const ov::Shape& s1, const ov::Shape& s2) {
    OPENVINO_ASSERT(s1.size() == s2.size(), "[GPU] Rank mismatch: ", s1, " and ", s2);

    std::vector<size_t> result;
    for (size_t i = 0; i < s1.size(); i++) {
        result.push_back(s1[i] * s2[i]);
    }
    return result;
}

static ov::Shape operator+(const ov::Shape& s1, const ov::Shape& s2) {
    OPENVINO_ASSERT(s1.size() == s2.size(), "[GPU] Rank mismatch: ", s1, " and ", s2);

    std::vector<size_t> result;
    for (size_t i = 0; i < s1.size(); i++)
        result.push_back(s1[i] + s2[i]);

    return result;
}

static ov::Shape operator-(const ov::Shape& s1, const ov::Shape& s2) {
    OPENVINO_ASSERT(s1.size() == s2.size(), "[GPU] Rank mismatch: ", s1, " and ", s2);

    std::vector<size_t> result;
    for (size_t i = 0; i < s1.size(); i++) {
        if (s1[i] < s2[i])
            return std::vector<size_t>();

        result.push_back(s1[i] - s2[i]);
    }

    return result;
}

void ShapePredictor::add_shape(const std::string& id, const ov::Shape& shape) {
    auto& shapes = _shapes_info[id];
    if (shapes.size() >= _max_deque_size)
        shapes.pop_front();

    shapes.push_back(shape);
}

bool ShapePredictor::can_preallocate(size_t desired_buffer_size) {
    const auto memory_threshold = 0.90f;
    auto device_mem_usage = _engine->get_used_device_memory(cldnn::allocation_type::usm_device);

    return device_mem_usage + desired_buffer_size < _engine->get_device_info().max_global_mem_size * memory_threshold;
}

std::pair<bool, ov::Shape> ShapePredictor::predict_preallocation_shape(const std::string& id,
                                                                       const ov::Shape& current_shape,
                                                                       size_t dt_size,
                                                                       bool can_reuse_buffer) {
    add_shape(id, current_shape);

    // Save shape information and exit without pre-allocation suggestion if current
    // buffer can be reused
    if (can_reuse_buffer)
        return {false, {}};

    // Check if there is enough data for prediction
    auto& shapes = _shapes_info[id];
    const auto shapes_num = shapes.size();

    // Number of shapes used for iterations mode predictions
    const auto min_shapes_num = _max_deque_size;

    if (shapes_num >= min_shapes_num) {
        std::vector<ov::Shape> diffs;

        for (size_t i = 0; i < min_shapes_num - 1; ++i) {
            auto result = shapes[shapes_num - i - 1] - shapes[shapes_num - i - 2];
            if (result.empty())
                break;
            diffs.push_back(result);
        }

        bool can_use_iterations_preallocation = diffs.size() == min_shapes_num - 1;
        for (size_t i = 1; i < diffs.size(); ++i) {
            if (diffs[0] != diffs[i]) {
                can_use_iterations_preallocation = false;
                break;
            }
        }

        if (can_use_iterations_preallocation)
            can_use_iterations_preallocation = !all_zeroes(diffs[0]);

        // Allow iterations preallocation only for per-dimension diff less than
        // '_max_per_dim_diff' and iteration size less than `_max_per_iter_size`
        // to avoid huge unexpected memory preallocations
        if (can_use_iterations_preallocation) {
            for (size_t i = 0; i < diffs[0].size(); ++i) {
                if (diffs[0][i] > _max_per_dim_diff) {
                    can_use_iterations_preallocation = false;
                    break;
                }
            }

            ov::Shape single_iter_shape;
            for (size_t i = 0; i < current_shape.size(); ++i)
                single_iter_shape.push_back(diffs[0][i] == 0 ? current_shape[i] : 1);

            if (ov::shape_size(single_iter_shape) * dt_size > _max_per_iter_size)
                can_use_iterations_preallocation = false;
        }

        if (can_use_iterations_preallocation) {
            // Apply preallocation for the next N iterations
            ov::Shape mul_shape(diffs[0].size(), _next_iters_preallocation_count);
            auto preallocation_shape = diffs[0] * mul_shape;
            auto new_shape = current_shape + preallocation_shape;
            return {true, new_shape};
        } else if (_buffers_preallocation_ratio > 1.0f) {
            // Apply percentage buffer preallocation
            auto current_shape_size = ov::shape_size(current_shape);
            ov::Shape new_shape_size(current_shape.size(), 1);
            new_shape_size[0] = static_cast<size_t>(current_shape_size * _buffers_preallocation_ratio);
            return {true, new_shape_size};
        }
    }

    return {false, {}};
}

}  // namespace cldnn
