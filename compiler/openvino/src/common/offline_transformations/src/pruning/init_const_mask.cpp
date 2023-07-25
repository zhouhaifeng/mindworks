// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/coordinate_transform.hpp>
#include <ngraph/log.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "mask_attribute.hpp"
#include "pruning.hpp"

ngraph::pass::InitConstMask::InitConstMask(const ngraph::AxisSet& dims,
                                           const std::function<bool(const double& value)>& condition) {
    auto constant = pattern::wrap_type<opset6::Constant>(
        pattern::type_matches_any({element::i8, element::u8, element::f16, element::f32, element::f64}));

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto const_node = std::dynamic_pointer_cast<opset6::Constant>(m.get_match_root());
        if (!const_node)
            return false;

        const auto& shape = const_node->get_shape();
        const auto& values = const_node->cast_vector<double>();

        auto mask = std::make_shared<Mask>(shape);

        for (const auto& dim : dims) {
            if (dim >= shape.size()) {
                OPENVINO_DEBUG << "[WARNING] Attemt to initialize masks on " << dim
                               << " dimension which is out of shape " << shape << " for node ("
                               << const_node->get_friendly_name() << ")";
                continue;
            }

            for (size_t value = 0; value < shape[dim]; ++value) {
                Coordinate begin(shape.size(), 0);
                Coordinate end(shape);

                begin[dim] = value;
                end[dim] = value + 1;

                bool skip_dim_value = false;
                NGRAPH_SUPPRESS_DEPRECATED_START
                CoordinateTransform iter(shape, begin, end);
                for (const Coordinate& coord : iter) {
                    if (!condition(values.at(iter.index(coord)))) {
                        skip_dim_value = true;
                        break;
                    }
                }
                NGRAPH_SUPPRESS_DEPRECATED_END
                if (!skip_dim_value) {
                    mask->at(dim).insert(value);
                }
            }
        }

        setMask(const_node, mask);
#ifdef ENABLE_OPENVINO_DEBUG
        setInitMask(const_node, mask);
#endif
        if (!mask->all_dims_are_empty()) {
            OPENVINO_DEBUG << "MASK (" << const_node->get_friendly_name() << ") " << *mask << std::endl;
        }

        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(constant, "InitConstMask");
    register_matcher(m, callback);
}
