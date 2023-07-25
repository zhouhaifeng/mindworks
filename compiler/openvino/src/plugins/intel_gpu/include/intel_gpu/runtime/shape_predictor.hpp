// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"

#include <deque>

namespace cldnn {

class engine;

struct ShapePredictor {
public:
    ShapePredictor(const engine* engine, float buffers_preallocation_ratio)
        : _engine(engine)
        , _buffers_preallocation_ratio(buffers_preallocation_ratio) {
        static_assert(_max_deque_size >= 2, "[GPU] Deque is supposed to contain at least 2 elements for prediction");
    }

    ShapePredictor(const engine* engine,
                   size_t next_iters_preallocation_count,
                   size_t max_per_iter_size,
                   size_t max_per_dim_diff,
                   float buffers_preallocation_ratio)
        : _engine(engine)
        , _next_iters_preallocation_count(next_iters_preallocation_count)
        , _max_per_iter_size(max_per_iter_size)
        , _max_per_dim_diff(max_per_dim_diff)
        , _buffers_preallocation_ratio(buffers_preallocation_ratio) {
        static_assert(_max_deque_size >= 2, "[GPU] Deque is supposed to contain at least 2 elements for prediction");
    }


/// \brief Predicts the next possible shapes sizes based on history collected by previous
///        predict_preallocation_shape() calls.
///        This function works in two modes: by default it tries to predict shape for the next
///        `_next_iters_preallocation_count` iterations, in case if per-iteration buffer size is less than
///        `_max_per_iter_size` and difference between shapes is less than `_max_per_dim_diff`; the second
///        operation mode is percentage preallocation - this mode can be configured with
///        ov::intel_gpu::buffers_preallocation_ratio property, it increases buffer size by
///        `_buffers_preallocation_ratio` value unconditionally.
/// \param id Primitive id.
/// \param current_shape Primitive's shape on current iteration.
/// \param dt_size Primitive's data_type size.
/// \param can_reuse_buffer Specifies if current memory buffer is enough to store data.
/// \return The result of shape size prediction as std::pair<bool, ov::Shape>, where the first element
///         says if shape is successfully predicted and can be preallocated, and the second element is ov::Shape itself.
    std::pair<bool, ov::Shape> predict_preallocation_shape(const std::string& id,
                                                           const ov::Shape& current_shape,
                                                           size_t dt_size,
                                                           bool can_reuse_buffer);
    bool can_preallocate(size_t desired_buffer_size);

private:
    void add_shape(const std::string& id, const ov::Shape& shape);

    static constexpr size_t _max_deque_size = 3;
    std::map<std::string, std::deque<ov::Shape>> _shapes_info;
    const engine* _engine;

    // Iterations mode preallocation
    const size_t _next_iters_preallocation_count = 10;
    const size_t _max_per_iter_size = 16 * 1024; // 16KB => maximum preallocation size is 16KB * 10iters = 160KB
    const size_t _max_per_dim_diff = 2;

    // Percentage mode preallocation
    const float _buffers_preallocation_ratio = 1.0f;
};

}  // namespace cldnn
