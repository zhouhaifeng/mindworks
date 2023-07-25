﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/fake_quantize.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"
#include "low_precision/rt_info/skip_cleanup_attribute.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FuseSubtractToFakeQuantizeTransformation::FuseSubtractToFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FuseSubtractToFakeQuantizeTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Subtract>();

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

bool FuseSubtractToFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    const auto subtract = m.get_match_root();
    if (!canBeTransformed(context, subtract)) {
        return false;
    }

    const auto parent = subtract->get_input_node_shared_ptr(0);
    auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(parent);
    const auto convert = ov::as_type_ptr<ov::opset1::Convert>(parent);

    if (convert) {
        fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    const auto subtractConstant = subtract->get_input_node_shared_ptr(1);
    if (!ov::is_type<ov::opset1::Constant>(subtractConstant)) {
        return false;
    }

    auto outputLowConst_f32 = foldConvert(fakeQuantize->input_value(3), deqPrecision);
    auto outputHighConst_f32 = foldConvert(fakeQuantize->input_value(4), deqPrecision);

    const auto value = subtractConstant->get_output_element_type(0) == element::f32 ?
        subtractConstant :
        foldConvert(subtractConstant, deqPrecision);

    outputLowConst_f32 = fold<ov::opset1::Subtract>(outputLowConst_f32, value);
    outputHighConst_f32 = fold<ov::opset1::Subtract>(outputHighConst_f32, value);

    const auto inputLow = foldConvert(fakeQuantize->input_value(1), deqPrecision);
    const auto inputHigh = foldConvert(fakeQuantize->input_value(2), deqPrecision);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(1), inputLow);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(2), inputHigh);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(3), outputLowConst_f32);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(4), outputHighConst_f32);

    auto newFakeQuantize = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
        ov::opset1::FakeQuantize(
            fakeQuantize->input_value(0),
            inputLow,
            inputHigh,
            outputLowConst_f32,
            outputHighConst_f32,
            fakeQuantize->get_levels()),
        subtract->get_output_element_type(0));

    replace_node(subtract, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);

    updateOutput(context, newFakeQuantize, subtract);
    return true;
}

bool FuseSubtractToFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!ov::is_type<ov::opset1::Constant>(operation->get_input_node_shared_ptr(1))) {
        return false;
    }

    if (!FakeQuantizeTransformation::checkElementwise(operation)) {
        return false;
    }

    if (!getAttribute<SkipCleanupAttribute>(operation).empty()) {
        return false;
    }

    const auto children = operation->get_output_target_inputs(0);

    for (const auto& target : children) {
        const auto convolution = ov::is_type<ov::opset1::Convolution>(target.get_node());
        const auto groupConvolution = ov::is_type<ov::opset1::GroupConvolution>(target.get_node());
        const auto convolutionBackpropData = ov::is_type<ov::opset1::ConvolutionBackpropData>(target.get_node());
        if (convolution || groupConvolution || convolutionBackpropData) {
            return false;
        }
    }

    const auto parent = operation->get_input_node_shared_ptr(0);
    auto fq = ov::as_type_ptr<ov::opset1::FakeQuantize>(parent);
    const auto convert = ov::as_type_ptr<ov::opset1::Convert>(parent);

    if (convert) {
        fq = ov::as_type_ptr<ov::opset1::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    if (!fq) {
        return false;
    }

    if (fq->get_output_target_inputs(0).size() != 1) {
        return false;
    }

    return true;
}

bool FuseSubtractToFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
