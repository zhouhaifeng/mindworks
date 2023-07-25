// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/smart_reshape/reshape_to_1D.hpp>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"

ov::pass::ReshapeTo1D::ReshapeTo1D() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(ReshapeTo1D);
    auto reshape_label = ngraph::pattern::wrap_type<ov::op::v1::Reshape>(
        {pattern::any_input(), ngraph::pattern::wrap_type<ov::op::v0::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });

    matcher_pass_callback callback = [](pattern::Matcher& m) -> bool {
        m.get_match_root()->input(1).replace_source_output(
            ov::op::v0::Constant::create(ngraph::element::i64, {1}, {-1}));
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_label /*, matcher_name*/);
    register_matcher(m, callback);
}
