// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/cascade_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string CascadeConcat::getTestCaseName(const testing::TestParamInfo<CascadeConcatTuple> &obj) {
    std::vector<std::vector<size_t>> input1, input2, input3;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    bool multioutput;
    std::map<std::string, std::string> additional_config;
    std::tie(input1, input2, input3, netPrecision, multioutput, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(input1[0]) << "_";
    results << CommonTestUtils::vec2str(input2[0]) << "_";
    results << CommonTestUtils::vec2str(input3[0]) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "Multioutput=" << multioutput << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void CascadeConcat::SetUp() {
    std::vector<std::vector<size_t>> input1, input2, input3;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    bool multioutput;
    std::tie(input1, input2, input3, netPrecision, multioutput, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto input = ngraph::builder::makeParams(ngPrc, {input1[0], input2[0], input3[0]});
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(input[0]);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(input[1]);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(input[2]);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0),
                                                                                relu2->output(0)},
                                                                                1);
    auto reshape = ngraph::builder::makeSqueezeUnsqueeze(concat, ngraph::element::i64, {0}, ngraph::helpers::SqueezeOpType::UNSQUEEZE);
    auto reshape2 = ngraph::builder::makeSqueezeUnsqueeze(reshape, ngraph::element::i64, {0}, ngraph::helpers::SqueezeOpType::SQUEEZE);
    auto concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{reshape2->output(0),
                                                                                 relu3->output(0)},
                                                                                 1);
    ngraph::ResultVector results;
    if (multioutput) {
        auto const_mult = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1, input1[0][1]+input2[0][1]},
                                                  std::vector<float>{1.01f});
        auto mult = std::make_shared<ngraph::op::v1::Multiply>(concat, const_mult);
        results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(concat2),
                                       std::make_shared<ngraph::opset1::Result>(mult)};
    } else {
        results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(concat2)};
    }
    function = std::make_shared<ngraph::Function>(results, input, "concat_reshape_reshape_concat_mul");
}

std::string CascadeConcatWithMultiConnReshape::getTestCaseName(const testing::TestParamInfo<CascadeConcatWithMultiConnReshapeTuple> &obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShape, netPrecision, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    for (auto const& configItem : additional_config) {
        results << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return results.str();
}

/**
 * Tests a case when 2 concats have Squeeze between them and Concat2 is the second connection of Squeeze output
 * Input     Const1
 *   |         |
 *  Relu       |
 *    |        |
 *      Concat1
 *        |
 *      Squeeze   Const2
 *    |        |   |
 *   Relu1    Concat2
 *    |          |
 * Unsqueeze1   Relu2
 *               |
 *            Unsqueeze2
 */
void CascadeConcatWithMultiConnReshape::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShape, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto inputShapeSqueezed = inputShape;
    inputShapeSqueezed.insert(std::begin(inputShapeSqueezed), 1);
    auto input = ngraph::builder::makeParams(ngPrc, {inputShapeSqueezed});
    auto relu = std::make_shared<ngraph::opset8::Relu>(input[0]);
    auto const1 = ngraph::builder::makeConstant(ngPrc, inputShapeSqueezed, std::vector<float>{}, true);
    auto concat1 = ngraph::builder::makeConcat({relu, const1}, inputShapeSqueezed.size() - 1);

    auto squeeze = ngraph::builder::makeSqueezeUnsqueeze(concat1, ngraph::element::i64, {0}, ngraph::helpers::SqueezeOpType::SQUEEZE);

    auto relu1 = std::make_shared<ngraph::opset8::Relu>(squeeze);
    auto unsqueeze1 = ngraph::builder::makeSqueezeUnsqueeze(relu1, ngraph::element::i64, {0}, ngraph::helpers::SqueezeOpType::UNSQUEEZE);

    auto const2 = ngraph::builder::makeConstant(ngPrc, inputShape, std::vector<float>{}, true);
    auto concat2 = ngraph::builder::makeConcat({squeeze, const2}, 1);
    // Change concat name to make it the second connection in the map of squeeze output connections
    concat2->set_friendly_name("XConcat");

    auto relu2 = std::make_shared<ngraph::opset8::Relu>(concat2);
    auto unsqueeze2 = ngraph::builder::makeSqueezeUnsqueeze(relu2, ngraph::element::i64, {0}, ngraph::helpers::SqueezeOpType::UNSQUEEZE);
    ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(unsqueeze1),
                                    std::make_shared<ngraph::opset1::Result>(unsqueeze2)};

    function = std::make_shared<ngraph::Function>(results, input, "CascadeConcatWithMultiConnReshapeTest");
}
} // namespace SubgraphTestsDefinitions
