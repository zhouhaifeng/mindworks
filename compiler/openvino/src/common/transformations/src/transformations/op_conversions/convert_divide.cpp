// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_divide.hpp"

#include <memory>
#include <ngraph/log.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/util/log.hpp"
#include "transformations/rt_info/nonconvertible_divide.hpp"
#include "transformations/utils/utils.hpp"

namespace {
bool convert_divide(std::shared_ptr<ngraph::Node> node) {
    auto div = std::dynamic_pointer_cast<ov::op::v1::Divide>(node);
    // We can not apply this transformation in case with integer input data type
    if (!div || ov::divide_is_nonconvertible(div) || div->get_input_element_type(0).is_integral()) {
        return false;
    }

    std::shared_ptr<ngraph::Node> pow = std::make_shared<ov::op::v1::Power>(
        div->input_value(1),
        ngraph::op::Constant::create(div->get_input_element_type(1), ngraph::Shape{}, {-1}));

    if (std::dynamic_pointer_cast<ngraph::op::Constant>(div->get_input_node_shared_ptr(1))) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        if (auto const_pow = ngraph::get_constant_from_source(pow)) {
            pow = const_pow;
        } else {
            OPENVINO_DEBUG << "ConvertDivide has failed due to unsupported evaluate type in " << pow.get();
            return false;
        }
        OPENVINO_SUPPRESS_DEPRECATED_END
    } else {
        ngraph::copy_runtime_info(div, pow);
    }

    auto mul = std::make_shared<ov::op::v1::Multiply>(div->input(0).get_source_output(), pow);
    // if Divide is an inverse, then we don't need the Multiply
    if (ov::op::util::can_eliminate_eltwise_node(mul, mul->input_value(0), mul->input_value(1))) {
        pow->set_friendly_name(div->get_friendly_name());
        ngraph::replace_node(div, pow);
    } else {
        mul->set_friendly_name(div->get_friendly_name());
        ngraph::copy_runtime_info(div, mul);
        ngraph::replace_node(div, mul);
    }
    return true;
}
}  // namespace

ov::pass::ConvertDivide::ConvertDivide() {
    MATCHER_SCOPE(ConvertDivide);
    auto div = ngraph::pattern::wrap_type<ov::op::v1::Divide>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        return convert_divide(m.get_match_root());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::ConvertDivideWithConstant::ConvertDivideWithConstant() {
    MATCHER_SCOPE(ConvertDivideWithConstant);
    auto div = ngraph::pattern::wrap_type<ov::op::v1::Divide>(
        {pattern::any_input(), pattern::wrap_type<ov::op::v0::Constant>()});

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        return convert_divide(m.get_match_root());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, matcher_name);
    this->register_matcher(m, callback);
}
