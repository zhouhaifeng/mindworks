// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "behavior/ov_infer_request/inference_chaining.hpp"

using namespace ov::test::behavior;

namespace {

auto configs = []() {
    return std::vector<ov::AnyMap>{{}};
};

auto AutoConfigs = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(CommonTestUtils::DEVICE_GPU, CommonTestUtils::DEVICE_CPU)},
                                   {}};
};

auto AutoNotSupportConfigs = []() {
    return std::vector<ov::AnyMap>{};
};

std::shared_ptr<ngraph::Function> getFunction1() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32;

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto relu = std::make_shared<ngraph::opset1::Relu>(params[0]);
    relu->get_output_tensor(0).set_names({"relu"});

    return std::make_shared<ngraph::Function>(relu, params, "SimpleActivation");
}

std::shared_ptr<ngraph::Function> getFunction2() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32;

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto in2add = ngraph::builder::makeConstant(ngPrc, {1, 2, 1, 1}, std::vector<float>{}, true);
    auto add = ngraph::builder::makeEltwise(split->output(0), in2add, ngraph::helpers::EltwiseTypes::ADD);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(add);

    auto in2mult = ngraph::builder::makeConstant(ngPrc, {1, 2, 1, 1}, std::vector<float>{}, true);
    auto mult = ngraph::builder::makeEltwise(split->output(1), in2mult, ngraph::helpers::EltwiseTypes::MULTIPLY);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(mult);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 3);
    concat->get_output_tensor(0).set_names({"concat"});

    return std::make_shared<ngraph::Function>(concat, params, "SplitAddConcat");
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_1, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction1()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 4, 20, 20}},
                                    {{2, 4, 20, 20}, {2, 4, 20, 20}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoConfigs())),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoConfigs())),
                        OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferenceChainingStatic,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoConfigs())),
                        OVInferenceChainingStatic::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVNotSupportRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(getFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoNotSupportConfigs())),
                        OVInferRequestDynamicTests::getTestCaseName);
}  // namespace
