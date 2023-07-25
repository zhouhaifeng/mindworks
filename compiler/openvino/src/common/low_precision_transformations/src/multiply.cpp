﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/multiply.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

MultiplyTransformation::MultiplyTransformation(const Params& params) : EltwiseBaseTransformation(params) {
    MATCHER_SCOPE(MultiplyTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Multiply>();

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

bool MultiplyTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(multiply, defaultPrecisions, 0));
    NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(multiply, defaultPrecisions, 1));

    multiply = NetworkHelper::separateInStandaloneBranch(multiply, defaultPrecisions);
    auto newMultiply = multiply;

    auto fold_fake_quantizes = [](std::shared_ptr<Node>& multiply, const size_t index) {
        auto fakeQuantizeOnWeights = ov::as_type_ptr<ov::opset1::FakeQuantize>(multiply->get_input_node_shared_ptr(index));
        if (fakeQuantizeOnWeights != nullptr) {
            auto result = NetworkHelper::fold_fake_quantize(fakeQuantizeOnWeights);
            if (ov::is_type<ov::opset1::Constant>(result)) {
                replace_node(fakeQuantizeOnWeights, result);
            }
        }
    };

    fold_fake_quantizes(multiply, 0ul);
    fold_fake_quantizes(multiply, 1ul);

    const int fullPathIndex = getNotEmpty(multiply);
    if (fullPathIndex == -1) {
        const auto multiplyBranch = getMultiplyConstBranch(multiply);
        if (multiplyBranch.first != -1) {
            NetworkHelper::foldDequantization(multiply, multiplyBranch.first == 0 ? 1 : 0, defaultPrecisions);
        }

        if (multiplyBranch.first == -1 || multiplyBranch.second == -1) {
            // constant folding on dequantization ops (for example: Convert on Subtract)
            NetworkHelper::foldDequantization(multiply, 0, defaultPrecisions);
            NetworkHelper::foldDequantization(multiply, 1, defaultPrecisions);
            return false;
        }

        auto multiplyParent = multiply->input_value(multiplyBranch.first);
        auto constParent = multiply->input_value(multiplyBranch.first == 0 ? 1 : 0);
        auto multiplyParentParent = multiplyParent.get_node_shared_ptr()->input_value(multiplyBranch.second);
        auto multiplyParentConst = multiplyParent.get_node_shared_ptr()->input_value(multiplyBranch.second == 0 ? 1 : 0);

        newMultiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
            std::vector<ngraph::element::Type>{ element::f32, element::f32 },
            std::vector<ngraph::element::Type>{ multiply->get_output_element_type(0) },
            ov::op::TemporaryReplaceOutputType(multiplyParentParent, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(
                fold<ov::opset1::Multiply>(
                    foldConvert(multiplyParentConst, element::f32),
                    foldConvert(constParent, element::f32)),
                element::f32).get());

        NetworkHelper::copyInfo(multiplyParent.get_node_shared_ptr(), newMultiply);
        NetworkHelper::copyInfo(multiply, newMultiply);
    } else {
        const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

        if (updatePrecisions) {
            const FakeQuantizeDequantization dequantizationEmptyPath = NetworkHelper::getDequantization(multiply, defaultPrecisions, emptyPathIndex);
            if (!dequantizationEmptyPath.empty() && !dequantizationEmptyPath.isLowPrecision()) {
                return false;
            }
        }

        FakeQuantizeDequantization dequantizationEmptyPath = NetworkHelper::foldDequantization(multiply, emptyPathIndex, defaultPrecisions);
        std::shared_ptr<Node> subtractValuesEmptyPath;
        std::shared_ptr<Node> multiplyValuesEmptyPath;
        std::tie(subtractValuesEmptyPath, multiplyValuesEmptyPath) = NetworkHelper::createEmptyValues(dequantizationEmptyPath, deqPrecision);

        // check if empty path shifts are not zero
        if (!NetworkHelper::isZeroConst(subtractValuesEmptyPath)) {
            return false;
        }

        FakeQuantizeDequantization dequantizationFullPath = NetworkHelper::foldDequantization(multiply, fullPathIndex, defaultPrecisions);
        std::shared_ptr<Node> subtractValuesFullPath;
        std::shared_ptr<Node> multiplyValuesFullPath;
        std::tie(subtractValuesFullPath, multiplyValuesFullPath) = NetworkHelper::createEmptyValues(dequantizationFullPath, deqPrecision);


        // before: Y = (SC1 * (X1 - SH1)) * (SC2 * X2)
        // after : Y = (SC1' * (X1 - SH1)) * (X2) , where :
        //         SC1' = SC1 * SC2
        auto newMultiplyValuesFullPath = fold<ov::opset1::Multiply>(multiplyValuesEmptyPath, multiplyValuesFullPath);
        OutputVector inputs{ {}, {} };
        inputs[emptyPathIndex] = dequantizationEmptyPath.data;
        inputs[fullPathIndex] = std::make_shared<ov::opset1::Multiply>(
            dequantizationFullPath.subtract == nullptr ?
                (dequantizationFullPath.convert == nullptr ?
                    dequantizationFullPath.data : dequantizationFullPath.convert) :
                dequantizationFullPath.subtract,
            newMultiplyValuesFullPath);

        newMultiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{ multiply->get_output_element_type(0) },
                ov::op::TemporaryReplaceOutputType(inputs[0], element::f32).get(),
                ov::op::TemporaryReplaceOutputType(inputs[1], element::f32).get());
        NetworkHelper::copyInfo(multiply, newMultiply);
    }

    replace_node(multiply, newMultiply);
    updateOutput(context, newMultiply, multiply);

    if (fullPathIndex != -1) {
        NetworkHelper::foldDequantization(newMultiply, fullPathIndex, defaultPrecisions);
    }

    return true;
}

bool MultiplyTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    FakeQuantizeDequantization dequantization1 = pass::low_precision::NetworkHelper::getDequantization(layer, defaultPrecisions, 0ul);
    FakeQuantizeDequantization dequantization2 = pass::low_precision::NetworkHelper::getDequantization(layer, defaultPrecisions, 1ul);

    if (dequantization1.data.get_node() == nullptr || dequantization2.data.get_node() == nullptr) {
        return false;
    }

    const bool nonConstantData = !ov::is_type<ov::opset1::Constant>(dequantization1.data.get_node_shared_ptr()) &&
                                 !ov::is_type<ov::opset1::Constant>(dequantization2.data.get_node_shared_ptr());

    if (((dequantization1.empty() || dequantization2.empty()) && nonConstantData)) {
        return false;
    }

    return EltwiseBaseTransformation::canBeTransformed(context, layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
