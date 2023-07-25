// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/smart_reshape/proposal_scales_stridedslice.hpp>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/proposal.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/strided_slice.hpp"

namespace {

bool crop_scales_for_proposal(const ngraph::pattern::PatternValueMap& pattern_to_output,
                              const std::shared_ptr<ngraph::Node>& parameter_label,
                              const std::shared_ptr<ngraph::Node>& proposal_label) {
    const auto& parameter = pattern_to_output.at(parameter_label);
    const auto& proposal = pattern_to_output.at(proposal_label).get_node_shared_ptr();

    auto cropped_scales = std::make_shared<ov::op::v1::StridedSlice>(
        proposal->input_value(2),
        ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}),
        ov::op::v0::Constant::create(ngraph::element::i64,
                                     ngraph::Shape{1},
                                     {parameter.get_partial_shape()[1].get_length()}),
        ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}),
        std::vector<int64_t>{0},
        std::vector<int64_t>{0});

    proposal->input(2).replace_source_output(cropped_scales->output(0));
    return true;
}

}  // namespace

ov::pass::Proposal1Scales::Proposal1Scales() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(Proposal1Scales);
    auto parameter_label = ngraph::pattern::wrap_type<ov::op::v0::Parameter>([](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() &&
               (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });
    auto convert_label = ngraph::pattern::wrap_type<ov::op::v0::Convert>({parameter_label});
    auto param_or_convert =
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{parameter_label, convert_label});
    auto reshape_label = ngraph::pattern::wrap_type<ov::op::v1::Reshape>(
        {param_or_convert, ngraph::pattern::wrap_type<ov::op::v0::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });
    auto proposal_label =
        ngraph::pattern::wrap_type<ov::op::v0::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher& m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal_label /*, matcher_name */);
    register_matcher(m, callback);
}

ov::pass::Proposal4Scales::Proposal4Scales() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(Proposal4Scales);
    auto parameter_label = ngraph::pattern::wrap_type<ov::op::v0::Parameter>([](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() &&
               (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });
    auto convert_label = ngraph::pattern::wrap_type<ov::op::v0::Convert>({parameter_label});
    auto param_or_convert =
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{parameter_label, convert_label});
    auto reshape_label = ngraph::pattern::wrap_type<ov::op::v1::Reshape>(
        {param_or_convert, ngraph::pattern::wrap_type<ov::op::v0::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });
    auto proposal_label =
        ngraph::pattern::wrap_type<ov::op::v4::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher& m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal_label /*, matcher_name */);
    register_matcher(m, callback);
}
