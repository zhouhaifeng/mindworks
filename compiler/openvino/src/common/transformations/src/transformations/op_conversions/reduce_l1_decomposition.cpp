// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/reduce_l1_decomposition.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_sum.hpp"

ov::pass::ReduceL1Decomposition::ReduceL1Decomposition() {
    MATCHER_SCOPE(ReduceL1Decomposition);
    // decomposes ReduceL1 operations into ReduceSum(abs(x))
    auto reduce_l1 = ngraph::pattern::wrap_type<ov::op::v4::ReduceL1>();

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto reduce_l1_node =
            std::dynamic_pointer_cast<ov::op::v4::ReduceL1>(pattern_to_output.at(reduce_l1).get_node_shared_ptr());

        if (reduce_l1_node == nullptr || transformation_callback(reduce_l1_node)) {
            return false;
        }

        auto abs = std::make_shared<ov::op::v0::Abs>(reduce_l1_node->input_value(0));
        auto reduce_sum = register_new_node<ov::op::v1::ReduceSum>(abs,
                                                                   reduce_l1_node->input_value(1),
                                                                   reduce_l1_node->get_keep_dims());

        reduce_sum->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(reduce_l1_node, {abs, reduce_sum});
        ngraph::replace_node(m.get_match_root(), reduce_sum);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_l1, matcher_name);
    register_matcher(m, callback);
}
