// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_upgrade.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "openvino/op/gather.hpp"

using namespace std;
using namespace ov;

pass::ConvertGather1ToGather7::ConvertGather1ToGather7() {
    MATCHER_SCOPE(ConvertGather1ToGather7);

    auto gather_v1_pattern = pattern::wrap_type<ov::op::v1::Gather>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_v1_node = std::dynamic_pointer_cast<ov::op::v1::Gather>(m.get_match_root());
        if (!gather_v1_node)
            return false;

        auto gather_v7_node = make_shared<ov::op::v7::Gather>(gather_v1_node->input_value(0),
                                                              gather_v1_node->input_value(1),
                                                              gather_v1_node->input_value(2),
                                                              0);

        gather_v7_node->set_friendly_name(gather_v1_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_v1_node, gather_v7_node);
        ngraph::replace_node(gather_v1_node, gather_v7_node);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(gather_v1_pattern, matcher_name);
    register_matcher(m, callback);
}

pass::ConvertGather7ToGather8::ConvertGather7ToGather8() {
    MATCHER_SCOPE(ConvertGather7ToGather8);

    auto gather_v7_pattern = pattern::wrap_type<ov::op::v7::Gather>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_v7_node = std::dynamic_pointer_cast<ov::op::v7::Gather>(m.get_match_root());
        if (!gather_v7_node)
            return false;

        auto gather_v8_node = make_shared<ov::op::v8::Gather>(gather_v7_node->input_value(0),
                                                              gather_v7_node->input_value(1),
                                                              gather_v7_node->input_value(2),
                                                              gather_v7_node->get_batch_dims());

        gather_v8_node->set_friendly_name(gather_v7_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_v7_node, gather_v8_node);
        ngraph::replace_node(gather_v7_node, gather_v8_node);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(gather_v7_pattern, matcher_name);
    register_matcher(m, callback);
}
