// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ngraph/runtime/host_tensor.hpp>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void experimental_detectron_topk_rois(const T* input_rois,
                                      const T* input_probs,
                                      const Shape& input_rois_shape,
                                      const Shape& input_probs_shape,
                                      size_t max_rois,
                                      T* output_rois) {
    const size_t input_rois_num = input_rois_shape[0];
    const size_t top_rois_num = std::min(max_rois, input_rois_num);

    std::vector<size_t> idx(input_rois_num);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&input_probs](size_t i1, size_t i2) {
        return input_probs[i1] > input_probs[i2];
    });

    for (size_t i = 0; i < top_rois_num; ++i) {
        output_rois[0] = input_rois[4 * idx[i] + 0];
        output_rois[1] = input_rois[4 * idx[i] + 1];
        output_rois[2] = input_rois[4 * idx[i] + 2];
        output_rois[3] = input_rois[4 * idx[i] + 3];
        output_rois += 4;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
