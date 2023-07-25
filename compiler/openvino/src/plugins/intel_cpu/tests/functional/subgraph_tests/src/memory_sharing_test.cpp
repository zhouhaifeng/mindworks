// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/convolution_params.hpp"

using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

class EdgeWithSameNameInTwoModels : public ::testing::Test, public CPUTestsBase {};

TEST_F(EdgeWithSameNameInTwoModels, smoke_CompareWithRef) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const std::string targetDevice = CommonTestUtils::DEVICE_CPU;
    const ov::element::Type type(ov::element::Type_t::f32);
    const std::string convName("conv_name"), weightName("weight_name");
    const std::vector<size_t> kernel{3, 3};
    const std::vector<size_t> strides{1, 1};
    const std::vector<ptrdiff_t> padsBegin{0, 0};
    const std::vector<ptrdiff_t> padsEnd{0, 0};
    const std::vector<size_t> dilations{1, 1};
    const ngraph::op::PadType autoPad(ngraph::op::PadType::EXPLICIT);

    if (InferenceEngine::with_cpu_x86_avx512f()) {
        std::tie(inFmts, outFmts, priority, selectedType) = conv_avx512_2D;
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        std::tie(inFmts, outFmts, priority, selectedType) = conv_avx2_2D;
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        std::tie(inFmts, outFmts, priority, selectedType) = conv_sse42_2D;
    }

    // first model
    const std::vector<std::vector<size_t>> shapes1{{1, 16, 720, 1280}};
    auto params1 = ngraph::builder::makeParams(type, shapes1);
    const size_t convOutCh1 = 32;
    auto conv1 = ngraph::builder::makeConvolution(params1.front(), type, kernel, strides, padsBegin, padsEnd, dilations, autoPad, convOutCh1);
    conv1->set_friendly_name(convName);
    conv1->get_input_node_shared_ptr(1)->set_friendly_name(weightName);
    auto model1 = makeNgraphFunction(type, params1, conv1, "Model1");

    // second model
    const std::vector<std::vector<size_t>> shapes2{{1, 32, 24, 24}};
    auto params2 = ngraph::builder::makeParams(type, shapes2);
    const size_t convOutCh2 = 16;
    auto conv2 = ngraph::builder::makeConvolution(params2.front(), type, kernel, strides, padsBegin, padsEnd, dilations, autoPad, convOutCh2);
    conv2->set_friendly_name(convName);
    conv2->get_input_node_shared_ptr(1)->set_friendly_name(weightName);
    auto model2 = makeNgraphFunction(type, params2, conv2, "Model2");

    // model compilation
    std::map<std::string, ov::AnyMap> config;
    auto& device_config = config[targetDevice];
    device_config[targetDevice + "_THROUGHPUT_STREAMS"] = 4;

    ov::Core core;
    for (auto&& item : config) {
        core.set_property(item.first, item.second);
    }

    auto compiledModel1 = core.compile_model(model1, targetDevice);
    auto compiledModel2 = core.compile_model(model2, targetDevice);

    auto inferReq1 = compiledModel1.create_infer_request();
    auto inferReq2 = compiledModel2.create_infer_request();

    inferReq1.infer();
    inferReq2.infer();
}

} // namespace SubgraphTestsDefinitions
