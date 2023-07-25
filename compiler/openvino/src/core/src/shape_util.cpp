// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/shape_util.hpp"

#include <algorithm>

#include "shape_util.hpp"

using namespace ngraph;

template <>
PartialShape ngraph::project(const PartialShape& shape, const AxisSet& axes) {
    if (shape.rank().is_dynamic()) {
        return shape;
    } else {
        std::vector<Dimension> result_dims;

        for (int64_t i = 0; i < shape.rank().get_length(); i++) {
            if (axes.find(i) != axes.end()) {
                result_dims.push_back(shape[i]);
            }
        }

        return PartialShape(result_dims);
    }
}

template <>
PartialShape ngraph::reduce(const PartialShape& shape, const AxisSet& deleted_axes, bool keep_dims) {
    if (shape.rank().is_dynamic()) {
        return shape;
    } else {
        std::vector<Dimension> result_dims;

        for (int64_t i = 0; i < shape.rank().get_length(); i++) {
            if (deleted_axes.find(i) == deleted_axes.end()) {
                result_dims.push_back(shape[i]);
            } else {
                if (keep_dims)
                    result_dims.emplace_back(1);
            }
        }

        return result_dims;
    }
}

template <>
PartialShape ngraph::inject_pairs(const PartialShape& shape,
                                  std::vector<std::pair<size_t, Dimension>> new_axis_pos_value_pairs) {
    if (shape.rank().is_dynamic()) {
        return shape;
    } else {
        std::vector<Dimension> result_dims;

        size_t original_pos = 0;

        for (size_t result_pos = 0; result_pos < shape.rank().get_length() + new_axis_pos_value_pairs.size();
             result_pos++) {
            auto search_it = std::find_if(new_axis_pos_value_pairs.begin(),
                                          new_axis_pos_value_pairs.end(),
                                          [result_pos](std::pair<size_t, Dimension> p) {
                                              return p.first == result_pos;
                                          });

            if (search_it == new_axis_pos_value_pairs.end()) {
                result_dims.push_back(shape[original_pos++]);
            } else {
                result_dims.push_back(search_it->second);
            }
        }

        return PartialShape{result_dims};
    }
}

namespace ov {
namespace util {
Shape make_dynamic_shape() {
    return Shape{0, std::numeric_limits<size_t>::max()};
}

bool is_dynamic_shape(const Shape& s) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    static const auto dyn_shape = make_dynamic_shape();
    OPENVINO_SUPPRESS_DEPRECATED_END
    return s == dyn_shape;
}
}  // namespace util
}  // namespace ov
