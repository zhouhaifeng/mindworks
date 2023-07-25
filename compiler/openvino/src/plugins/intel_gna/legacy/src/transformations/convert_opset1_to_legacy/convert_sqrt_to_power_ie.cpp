// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_sqrt_to_power_ie.hpp"

#include <legacy/ngraph_ops/power.hpp>
#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

ngraph::pass::ConvertSqrtToPowerIEMatcher::ConvertSqrtToPowerIEMatcher() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto sqrt = std::make_shared<ngraph::opset1::Sqrt>(input_0);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto sqrt = std::dynamic_pointer_cast<ngraph::opset1::Sqrt>(m.get_match_root());
        if (!sqrt) {
            return false;
        }
        auto power_ie = std::make_shared<ngraph::op::PowerIE>(sqrt->input(0).get_source_output(),
                                                              0.5f,
                                                              1.0f,
                                                              0.0f,
                                                              sqrt->output(0).get_element_type());
        power_ie->set_friendly_name(sqrt->get_friendly_name());
        ngraph::copy_runtime_info(sqrt, power_ie);
        ngraph::replace_node(sqrt, power_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sqrt, "ConvertPowerToPowerIE");
    this->register_matcher(m, callback);
}
