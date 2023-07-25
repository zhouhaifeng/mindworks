// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"

ov::pass::EliminateUnsqueezeGather::EliminateUnsqueezeGather() {
    MATCHER_SCOPE(EliminateUnsqueezeGather);
    // Remove Unsqueeze + Gather pair, if Gather gathers data by `1` dimension that was previously added by Unsqueeze
    const auto unsqueezeAxis = pass::pattern::any_input();
    const auto unsqueezeInput = pass::pattern::any_input();
    const auto unsqueeze =
        ngraph::pattern::wrap_type<ov::op::v0::Unsqueeze>({unsqueezeInput, unsqueezeAxis}, pattern::consumers_count(1));
    const auto gatherIndices = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
    const auto gatherAxis = pass::pattern::any_input();
    const auto gather = ngraph::pattern::wrap_type<op::util::GatherBase>({unsqueeze, gatherIndices, gatherAxis});

    ov::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& patternValue = m.get_pattern_value_map();

        const auto& m_unsqueezeAxis = patternValue.at(unsqueezeAxis);
        const auto& m_gatherAxis = patternValue.at(gatherAxis);

        const auto& unsqueezeAxisNode =
            ngraph::as_type_ptr<ov::op::v0::Constant>(m_unsqueezeAxis.get_node_shared_ptr());
        const auto& gatherAxisNode = ngraph::as_type_ptr<ov::op::v0::Constant>(m_gatherAxis.get_node_shared_ptr());

        if (!unsqueezeAxisNode || !gatherAxisNode) {
            return false;
        }

        const auto& unsqueezeAxisVec = unsqueezeAxisNode->cast_vector<int64_t>();
        const auto& gatherAxisVec = gatherAxisNode->cast_vector<int64_t>();

        if (unsqueezeAxisVec.size() != 1 || gatherAxisVec.size() != 1) {
            return false;
        }

        if (unsqueezeAxisVec.front() != gatherAxisVec.front()) {
            return false;
        }

        auto& m_gather = patternValue.at(gather);
        const auto& m_unsqueeze = patternValue.at(unsqueeze);
        const auto& m_unsqueezeInput = patternValue.at(unsqueezeInput);

        ngraph::copy_runtime_info(m_gather.get_node_shared_ptr(), m_unsqueeze.get_node_shared_ptr());
        m_gather.replace(m_unsqueezeInput);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather, matcher_name);
    register_matcher(m, callback);
}

ov::pass::EliminateGatherUnsqueeze::EliminateGatherUnsqueeze() {
    MATCHER_SCOPE(EliminateGatherUnsqueeze);

    const auto gather_indices_label = ngraph::pattern::wrap_type<ov::op::v0::Constant>(pattern::rank_equals(0));
    const auto gather_axis_label = ngraph::pattern::wrap_type<ov::op::v0::Constant>();
    const auto gather_label = ngraph::pattern::wrap_type<op::util::GatherBase>(
        {pass::pattern::any_input(), gather_indices_label, gather_axis_label},
        pattern::rank_equals(0));

    const auto unsqueeze_label =
        ngraph::pattern::wrap_type<ov::op::v0::Unsqueeze>({gather_label, pass::pattern::any_input()},
                                                          pattern::rank_equals(1));

    ov::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto pattern_nodes = m.get_pattern_map();

        auto& gather_indices = pattern_nodes.at(gather_indices_label);
        auto& gather = pattern_nodes.at(gather_label);
        auto& unsqueeze = pattern_nodes.at(unsqueeze_label);

        auto new_indices =
            ov::op::util::make_try_fold<ov::op::v1::Reshape>(gather_indices,
                                                             ov::op::v0::Constant::create(element::i32, {1}, {1}),
                                                             false);
        auto new_gather = gather->clone_with_new_inputs({gather->input_value(0), new_indices, gather->input_value(2)});

        new_gather->set_friendly_name(gather->get_friendly_name());
        ngraph::copy_runtime_info({unsqueeze, gather}, {new_gather, new_indices});
        ngraph::replace_node(unsqueeze, new_gather);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(unsqueeze_label, matcher_name);
    register_matcher(m, callback);
}
