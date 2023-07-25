// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_reduce_multi_axis.hpp"

#include <ngraph/rt_info.hpp>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>

template <class T>
ngraph::matcher_pass_callback ov::intel_cpu::ConvertReduceMultiAxisBase::convert_reduce() {
    return [&](ngraph::pattern::Matcher& m) {
        auto reduce = std::dynamic_pointer_cast<T>(m.get_match_root());
        if (!reduce) {
            return false;
        }

        const auto& input0 = reduce->input_value(0);
        const auto& input1 = reduce->input_value(1);
        const auto& data_shape0 = input0.get_partial_shape();
        auto reduction_axes = std::dynamic_pointer_cast<ov::opset8::Constant>(input1.get_node_shared_ptr());
        if (!reduction_axes) {
            return false;
        }
        if (ngraph::shape_size(input1.get_shape()) <= 1) {
            return false;
        }

        auto axes = reduction_axes->template cast_vector<int64_t>();
        for (auto axis : axes) {
            if (data_shape0[axis].is_dynamic()) {
                return false;
            }
        }

        ngraph::NodeVector new_ops;
        std::shared_ptr<ngraph::Node> node = input0.get_node_shared_ptr();
        bool keepDims = reduce->get_keep_dims();
        //axes should be sorted in descending order if keepDims is false to be keep axis within data shape
        if (!keepDims) {
            sort(axes.begin(), axes.end(), std::greater<int64_t>());
        }
        for (auto axis : axes) {
            auto reduction_axis = ov::opset8::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {axis});
            node = std::make_shared<T>(node, reduction_axis, keepDims);
            new_ops.push_back(node);
        }

        node->set_friendly_name(reduce->get_friendly_name());
        ngraph::copy_runtime_info(reduce, new_ops);
        ngraph::replace_node(reduce, node);
        return true;
    };
}

ov::intel_cpu::ConvertReduceProd::ConvertReduceProd() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceProd>({ngraph::pattern::any_input(),
                                                           ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceProd");
    register_matcher(m, convert_reduce<ov::opset8::ReduceProd>());
}

ov::intel_cpu::ConvertReduceMin::ConvertReduceMin() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceMin>({ngraph::pattern::any_input(),
                                                          ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceMin");
    register_matcher(m, convert_reduce<ov::opset8::ReduceMin>());
}

ov::intel_cpu::ConvertReduceMax::ConvertReduceMax() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceMax>({ngraph::pattern::any_input(),
                                                          ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceMax");
    register_matcher(m, convert_reduce<ov::opset8::ReduceMax>());
}

ov::intel_cpu::ConvertReduceSum::ConvertReduceSum() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::ReduceSum>({ngraph::pattern::any_input(),
                                                          ngraph::pattern::wrap_type<ov::opset8::Constant>()}), "ConvertReduceSum");
    register_matcher(m, convert_reduce<ov::opset8::ReduceSum>());
}
