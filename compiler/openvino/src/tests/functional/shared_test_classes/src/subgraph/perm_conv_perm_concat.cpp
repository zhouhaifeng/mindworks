// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/perm_conv_perm_concat.hpp"

namespace SubgraphTestsDefinitions {
std::string PermConvPermConcat::getTestCaseName(const testing::TestParamInfo<PermConvPermConcatParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::array<size_t, 4> input_shape;
    std::array<size_t, 2> kernel_shape;
    size_t output_channels;
    std::map<std::string, std::string> configuration;


    std::tie(netPrecision, targetName, input_shape, kernel_shape, output_channels, configuration) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(std::vector<size_t>(input_shape.begin(), input_shape.end())) << "_";
    results << "KS=" << CommonTestUtils::vec2str(std::vector<size_t>(kernel_shape.begin(), kernel_shape.end())) << "_";
    results << "OC=" << output_channels << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName;
    for (auto const& configItem : configuration) {
        results << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return results.str();
}

void PermConvPermConcat::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::array<size_t, 4> input_shape;
    std::array<size_t, 2> kernel_shape;
    size_t output_channels;
    std::map<std::string, std::string> additional_config;

    std::tie(netPrecision, targetDevice, input_shape, kernel_shape, output_channels, additional_config) = this->GetParam();

    configuration.insert(additional_config.begin(), additional_config.end());

    const std::size_t input_dim = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, input_dim };
    std::vector<size_t> reshape_in_dims = std::vector<size_t>(input_shape.begin(), input_shape.end());
    std::vector<size_t> permute_in_order = { 0, 3, 1, 2 };
    std::vector<size_t> permute_out_order = { 0, 2, 3, 1 };

    auto input_parameter = ngraph::builder::makeParams(ngPrc, {input_dims});

    auto reshape_in_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
        ngraph::Shape{4},
        reshape_in_dims);
    auto reshape_in = std::make_shared<ngraph::op::v1::Reshape>(input_parameter[0], reshape_in_pattern, false);

    auto permute_in_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
        ngraph::Shape{4},
        ngraph::Shape{permute_in_order});
    auto permute_in = std::make_shared<ngraph::opset1::Transpose>(reshape_in, permute_in_params);
    auto conv_in_shape = permute_in->get_output_shape(0);
    auto conv_weights_size = output_channels * (conv_in_shape[1]) * kernel_shape[0] * kernel_shape[1];
    auto conv = ngraph::builder::makeConvolution(permute_in, ngPrc, {kernel_shape[0], kernel_shape[1]}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
        ngraph::op::PadType::VALID, output_channels, false, CommonTestUtils::generate_float_numbers(conv_weights_size, -0.5f, 0.5f));

    auto permute_out_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
        ngraph::Shape{4},
        permute_out_order);
    auto permute_out = std::make_shared<ngraph::opset1::Transpose>(conv, permute_out_params);

    auto permute_out_shape = permute_out->get_output_shape(0);

    auto concat_const = ngraph::builder::makeConstant(ngPrc, {1, 1, 1, permute_out_shape[3]},
                                                      CommonTestUtils::generate_float_numbers(permute_out_shape[3], -10, 10));

    auto concat = ngraph::builder::makeConcat({permute_out, concat_const}, 2);

    auto reshape_out_pattern = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
        ngraph::Shape{2},
        InferenceEngine::SizeVector({1, (permute_out_shape[2] + 1) * permute_out_shape[3]}));
    auto reshape_out = std::make_shared<ngraph::op::v1::Reshape>(concat, reshape_out_pattern, false);

    function = std::make_shared<ngraph::Function>(reshape_out, input_parameter, "perm_conv_perm_concat");
}

void PermConvPermConcat::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();

    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        auto tensorDesc = info->getTensorDesc();

        auto blob = FuncTestUtils::createAndFillBlobFloat(tensorDesc, 2, -1, 100, 111);

        FuncTestUtils::fillInputsBySinValues(blob);
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
    }
    inferRequest.Infer();

    Validate();
}
} // namespace SubgraphTestsDefinitions
