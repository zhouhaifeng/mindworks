﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convert.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ConvertTransformation::ConvertTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ConvertTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Convert>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ConvertTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<ov::opset1::Convert> convert = ov::as_type_ptr<ov::opset1::Convert>(m.get_match_root());
    if (!convert) {
        return false;
    }

    if (!canBeTransformed(context, convert)) {
        return false;
    }

    const ngraph::element::Type precisionBefore = convert->get_input_element_type(0);

    std::shared_ptr<ov::opset1::Subtract> subtract = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
        convert->input_value(0),
        std::make_shared<ov::opset1::Constant>(precisionBefore, Shape{}, std::vector<size_t>({ 0 })));
    NetworkHelper::setOutDataPrecision(subtract, convert->get_output_element_type(0));

    replace_node(convert, subtract);

    subtract->set_friendly_name(convert->get_friendly_name());
    return true;
}

bool ConvertTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
