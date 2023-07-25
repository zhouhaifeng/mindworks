// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"

ov::pass::SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto softmax = pattern::wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>();
    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto m_softmax = m.get_match_root();
        Output<Node> input;
        int64_t softmax_axis;

        if (transformation_callback(m_softmax)) {
            return false;
        }

        if (auto m_softmax_v1 = std::dynamic_pointer_cast<ov::op::v1::Softmax>(m_softmax)) {
            input = m_softmax_v1->input_value(0);
            softmax_axis = static_cast<int64_t>(m_softmax_v1->get_axis());
        } else if (auto m_softmax_v8 = std::dynamic_pointer_cast<ov::op::v8::Softmax>(m_softmax)) {
            input = m_softmax_v8->input_value(0);
            softmax_axis = m_softmax_v8->get_axis();
        } else {
            return false;
        }

        auto axis = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {softmax_axis});
        auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(input, axis, true);
        auto sub = std::make_shared<ov::op::v1::Subtract>(input, reduce_max);
        auto exp = std::make_shared<ov::op::v0::Exp>(sub);
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(exp, axis, true);
        auto div = std::make_shared<ov::op::v1::Divide>(exp, reduce_sum);

        replace_node(m_softmax, div);
        copy_runtime_info(m_softmax, {reduce_max, reduce_sum, sub, exp, div});
        div->set_friendly_name(m_softmax->get_friendly_name());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(softmax, matcher_name);
    register_matcher(m, callback);
}
