// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/softmax.hpp"
#include "subgraph_softmax.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Softmax::getTestCaseName(testing::TestParamInfo<ov::test::snippets::SoftmaxParams> obj) {
    ov::Shape inputShapes;
    int axis;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, axis, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Axis=" << axis << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Softmax::SetUp() {
    ov::Shape inputShape;
    int axis;
    std::tie(inputShape, axis, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape, }}});

    auto f = ov::test::snippets::SoftmaxFunction({inputShape}, axis);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

std::string AddSoftmax::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddSoftmaxParams> obj) {
    std::pair<ov::Shape, ov::Shape> inputShapes;
    int axis;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, axis, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes.first) << "_";
    result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
    result << "Axis=" << axis << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddSoftmax::SetUp() {
    std::pair<ov::Shape, ov::Shape> inputShapes;
    int axis;
    std::tie(inputShapes, axis, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShapes.first, }}, {{}, {inputShapes.second, }}});

    auto f = ov::test::snippets::AddSoftmaxFunction({inputShapes.first, inputShapes.second}, axis);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

TEST_P(Softmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(AddSoftmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
