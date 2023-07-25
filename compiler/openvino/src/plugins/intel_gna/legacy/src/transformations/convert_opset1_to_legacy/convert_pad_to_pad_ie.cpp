// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

ngraph::pass::ConvertPadToLegacyMatcher::ConvertPadToLegacyMatcher() {
    auto m_pad = ngraph::pattern::wrap_type<ngraph::opset1::Pad>(pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto pad = std::dynamic_pointer_cast<ngraph::opset1::Pad>(m.get_match_root());
        if (!pad) {
            return false;
        }

        auto pad_ie = std::make_shared<ngraph::op::PadIE>(pad);
        pad_ie->set_friendly_name(pad->get_friendly_name());
        ngraph::copy_runtime_info(pad, pad_ie);
        ngraph::replace_node(pad, pad_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_pad, "ConvertPadToLegacy");
    this->register_matcher(m, callback);
}
