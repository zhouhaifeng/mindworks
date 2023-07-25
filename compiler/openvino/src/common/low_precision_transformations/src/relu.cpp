﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/relu.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ReluTransformation::ReluTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ReluTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Relu>({ pattern::wrap_type<ov::opset1::Multiply>() });

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

bool ReluTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<Node> relu = m.get_match_root();
    if (!canBeTransformed(context, relu)) {
        return false;
    }

    relu = NetworkHelper::separateInStandaloneBranch(relu, defaultPrecisions);
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(relu, defaultPrecisions, 0);
    moveDequantizationAfter(context, relu, dequantization, false, false);
    return true;
}

bool ReluTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

bool ReluTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions, 0);
    if (dequantization.empty() || (dequantization.subtract != nullptr)) {
        return false;
    }

    const auto scales = dequantization.multiplyConstant->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.f; })) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
