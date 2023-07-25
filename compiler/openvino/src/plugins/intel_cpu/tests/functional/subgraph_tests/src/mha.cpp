// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ngraph_functions/builders.hpp>
#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace CPUTestUtils;
using namespace ov::test;
using namespace ngraph::helpers;

namespace CPUSubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,   // Input shapes
        std::vector<ElementType>,  // Input precisions
        std::vector<ElementType>,  // MatMul input #0 precisions
        size_t,                    // pattern type #
        std::string,               // Expected node
        std::string                // Device name
> MHATuple;

static std::shared_ptr<ov::Model> initMHASubgraph0(std::vector<ov::PartialShape>& inputDynamicShapes, std::vector<ElementType>& inputPrecisions) {
    ngraph::ParameterVector ngraphParam;

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    ngraphParam.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    ngraphParam.push_back(transpose1Param);

    auto addParam = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    ngraphParam.push_back(addParam);

    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    ngraphParam.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, inputDynamicShapes[0].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<float> mulConstData(ngraph::shape_size(constantShapes[2]));
    auto mulConst = ngraph::builder::makeConstant(inputPrecisions[0], constantShapes[2], mulConstData, true);

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0] *
                                                                   inputDynamicShapes[0].get_shape()[1] * inputDynamicShapes[0].get_shape()[2]),
                                             -1};
    auto reshape0Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[3], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[2]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[4], reshape1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[5], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[6], transpose3ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, mulConst);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ngraph::Function>(results, ngraphParam, "mha");
}

static std::shared_ptr<ov::Model> initMHASubgraph1(std::vector<ov::PartialShape>& inputDynamicShapes, std::vector<ElementType>& inputPrecisions) {
    ngraph::ParameterVector ngraphParam;

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    ngraphParam.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    ngraphParam.push_back(transpose1Param);

    auto addParam = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    ngraphParam.push_back(addParam);

    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    ngraphParam.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1, inputDynamicShapes[0].get_shape()[2], 1, 1}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[0], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[1], transpose3ConstData);

    float transA = false;
    float transB = false;
    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(transpose0Param, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1, transA, transB);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, addParam);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(add, 3);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(softMax, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(matMul1, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ngraph::Function>(results, ngraphParam, "mha");
}

class MHATest : public testing::WithParamInterface<MHATuple>,
                         virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MHATuple> &obj) {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        size_t patternType;
        std::string expectedNode;
        std::string targetName;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNode, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=(";
        for (const auto& shape : inputShapes) {
            results << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        results << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                results << CommonTestUtils::vec2str(item) << "_";
            }
        }
        for (size_t i = 0; i < inputPrecisions.size(); i++) {
            results << "InPRC" << std::to_string(i) << "=" << inputPrecisions[i] << "_";
        }
        results << "patternType=" << patternType;
        results << "expect=" << expectedNode;
        results << "targetDevice=" << targetName;

        return results.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            // TODO: after snippets fixed should remove 2nd condition, ticket: 105339
            if (patternType == 0 || expectedNode == "Subgraph")
                tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(funcInput.get_element_type(), targetInputStaticShapes[i], 1.0f, 0.5f);
            else
                // generate all negative inputs
                tensor = ov::test::utils::create_and_fill_tensor_unique_sequence(funcInput.get_element_type(), targetInputStaticShapes[i], -1, -5);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    size_t patternType;
    std::string expectedNode;
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNode, targetDevice) = this->GetParam();

        init_input_shapes(inputShapes);

        if (patternType == 0) {
            function = initMHASubgraph0(inputDynamicShapes, inputPrecisions);
        } else if (patternType == 1) {
            function = initMHASubgraph1(inputDynamicShapes, inputPrecisions);
        } else {
            FAIL() << "Unsupported MHA pattern type";
        }

        // TODO: try better input data initialization to avoid threshold adjustment
        // TODO: support different precisions on inputs
        if (inputPrecisions[0] == ElementType::bf16) {
            abs_threshold = 0.1f;
            rel_threshold = 10.f;

            configuration.insert({{ InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES }});
        }

        // Snippets MHA tokenization has limitations to avoid performance degradations. These limitations depend on target machine.
        // Just for testing, we disable these limitations to allow Snippets to tokenize pattern on all machines for validation.
        if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
            configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                                  InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
        }
    }
};

TEST_P(MHATest, CompareWithRefs) {
    std::vector<InputShape> inputShapes;
    std::vector<ElementType> inputPrecisions;
    std::vector<ElementType> matMulIn0Precisions;
    size_t patternType;
    std::string expectedNode;
    std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNode, targetDevice) = this->GetParam();

    if (inputPrecisions[0] == ElementType::bf16 && !InferenceEngine::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    if (!InferenceEngine::with_cpu_x86_avx512_core())
        GTEST_SKIP();

    run();
    CheckNumberOfNodesWithType(compiledModel, expectedNode, 1);
}

namespace {

std::vector<std::vector<ngraph::Shape>> inputShapes = {
    {{2, 8, 16, 64}, {2, 8, 16, 64}, {2, 1, 1, 8}, {2, 8, 16, 64}},
    {{1, 384, 16, 64}, {1, 384, 16, 64}, {1, 1, 1, 384}, {1, 384, 16, 64}},
    {{2, 64, 16, 80}, {2, 64, 16, 80}, {2, 1, 1, 64}, {2, 64, 16, 80}},
    {{3, 96, 16, 64}, {3, 96, 16, 64}, {3, 1, 1, 96}, {3, 96, 16, 64}},
    {{2, 192, 16, 160}, {2, 192, 16, 160}, {2, 1, 1, 192}, {2, 192, 16, 160}},
    {{2, 4, 16, 8}, {2, 4, 16, 8}, {2, 1, 1, 4}, {2, 4, 16, 8}},
    {{1, 204, 13, 212},  {1, 204, 13, 212},  {1, 1, 1, 204}, {1, 204, 13, 212}},
};

std::vector<std::vector<ElementType>> matMulIn0Precisions = {
    {},
};

std::vector<size_t> patternTypes = {
    0, 1
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHATest,
                        ::testing::Combine(
                                ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                ::testing::Values(std::vector<ElementType>{ ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32 }),
                                ::testing::ValuesIn(matMulIn0Precisions),
                                ::testing::ValuesIn(patternTypes),
                                ::testing::Values("Subgraph"),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MHATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MHA, MHATest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                 ::testing::Values(std::vector<ElementType>{ ElementType::bf16, ElementType::bf16, ElementType::bf16, ElementType::bf16 }),
                                 ::testing::ValuesIn(matMulIn0Precisions),
                                 ::testing::ValuesIn(patternTypes),
                                 ::testing::Values("MHA"),  // Snippets don't support BF16 MHA pattern yet
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHATest::getTestCaseName);

} // namespace

static std::shared_ptr<ov::Model> initMHAQuantSubgraph0(std::vector<ov::PartialShape>& inputDynamicShapes, std::vector<ElementType>& inputPrecisions,
                                                        std::vector<ElementType>& matMulIn0Precisions) {
    ngraph::ParameterVector ngraphParam;

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    ngraphParam.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    ngraphParam.push_back(transpose1Param);

    auto addParam = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    ngraphParam.push_back(addParam);

    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    ngraphParam.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({2}));
    constantShapes.push_back(ov::Shape({4}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<int64_t> reshape0ConstData = {static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0] *
                                                                   inputDynamicShapes[0].get_shape()[1] * inputDynamicShapes[0].get_shape()[2]),
                                             -1};
    auto reshape0Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[2], reshape0ConstData);

    std::vector<int64_t> reshape1ConstData = {static_cast<int64_t>(inputDynamicShapes[0].get_shape()[0]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[2]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1]),
                                              static_cast<int64_t>(inputDynamicShapes[0].get_shape()[1])};
    auto reshape1Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[3], reshape1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[4], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[5], transpose3ConstData);

    float transA = false;
    float transB = false;

    std::shared_ptr<ov::Node> fakeQuantize0;
    if (matMulIn0Precisions[0] == ElementType::u8)
        fakeQuantize0 = ngraph::builder::makeFakeQuantize(transpose0Param, inputPrecisions[0], 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
    else
        fakeQuantize0 = ngraph::builder::makeFakeQuantize(transpose0Param, inputPrecisions[0], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(transpose1Param, inputPrecisions[1], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(transpose2Param, inputPrecisions[3], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

    std::shared_ptr<ov::Node> fakeQuantize4;

    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize0, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize1, transpose1Const);
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1, transA, transB);
    const auto fakeQuantize3 = ngraph::builder::makeFakeQuantize(matMul0, inputPrecisions[0], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
    const auto add = std::make_shared<ngraph::opset3::Add>(fakeQuantize3, addParam);
    const auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, reshape0Const, true);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(reshape0, 1);
    const auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softMax, reshape1Const, true);
    if (matMulIn0Precisions[1] == ElementType::u8)
        fakeQuantize4 = ngraph::builder::makeFakeQuantize(reshape1, inputPrecisions[0], 256, {}, {0.0f}, {0.255f}, {0.0f}, {0.255f});
    else
        fakeQuantize4 = ngraph::builder::makeFakeQuantize(reshape1, inputPrecisions[0], 256, {}, {-0.128f}, {0.127f}, {-0.128f}, {0.127f});
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize2, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(fakeQuantize4, transpose2, transA, transB);
    const auto fakeQuantize5 = ngraph::builder::makeFakeQuantize(matMul1, inputPrecisions[0], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize5, transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ngraph::Function>(results, ngraphParam, "mha");
}

static std::shared_ptr<ov::Model> initMHAQuantSubgraph1(const std::vector<ov::PartialShape>& inputDynamicShapes,
                                                        const std::vector<ElementType>& inputPrecisions,
                                                        const std::vector<ElementType>& matMulIn0Precisions,
                                                        const bool fakeQuantize3Exists) {
    ngraph::ParameterVector ngraphParam;

    auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[0], inputDynamicShapes[0]);
    ngraphParam.push_back(transpose0Param);

    auto transpose1Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[1], inputDynamicShapes[1]);
    ngraphParam.push_back(transpose1Param);

    auto addParam = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[2], inputDynamicShapes[2]);
    ngraphParam.push_back(addParam);

    auto transpose2Param = std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[3], inputDynamicShapes[3]);
    ngraphParam.push_back(transpose2Param);

    std::vector<ov::Shape> constantShapes;
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({inputDynamicShapes[0].get_shape().size()}));
    constantShapes.push_back(ov::Shape({1}));

    std::vector<int64_t> transpose0ConstData = {0, 2, 1, 3};
    auto transpose0Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[0], transpose0ConstData);

    std::vector<int64_t> transpose1ConstData = {0, 2, 3, 1};
    auto transpose1Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[1], transpose1ConstData);

    std::vector<int64_t> transpose2ConstData = {0, 2, 1, 3};
    auto transpose2Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[2], transpose2ConstData);

    std::vector<int64_t> transpose3ConstData = {0, 2, 1, 3};
    auto transpose3Const = ngraph::builder::makeConstant(ElementType::i64, constantShapes[3], transpose3ConstData);

    std::vector<float> mulConstData(ngraph::shape_size(constantShapes[4]));
    auto mulConst = ngraph::builder::makeConstant(inputPrecisions[0], constantShapes[4], mulConstData, true);

    float transA = false;
    float transB = false;

    std::shared_ptr<ov::Node> fakeQuantize0;
    if (matMulIn0Precisions[0] == ElementType::u8)
        fakeQuantize0 = ngraph::builder::makeFakeQuantize(transpose0Param, inputPrecisions[0], 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
    else
        fakeQuantize0 = ngraph::builder::makeFakeQuantize(transpose0Param, inputPrecisions[0], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

    const auto transpose0 = std::make_shared<ov::op::v1::Transpose>(fakeQuantize0, transpose0Const);
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(transpose1Param, transpose1Const);
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(transpose1, inputPrecisions[1], 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
    const auto matMul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, fakeQuantize1, transA, transB);
    const auto mul = std::make_shared<ngraph::opset3::Multiply>(addParam, mulConst);
    const auto add = std::make_shared<ngraph::opset3::Add>(matMul0, mul);
    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(add, 3);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose2Param, transpose2Const);
    const auto matMul1 = std::make_shared<ngraph::opset3::MatMul>(softMax, transpose2, transA, transB);
    const auto transpose3 = std::make_shared<ov::op::v1::Transpose>(
        fakeQuantize3Exists ?
            ngraph::builder::makeFakeQuantize(matMul1, inputPrecisions[0], 256, {}, { 0.0f }, { 2.55f }, { 0.0f }, { 2.55f }) :
            matMul1,
        transpose3Const);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose3)};
    return std::make_shared<ngraph::Function>(results, ngraphParam, "mha");
}

class MHAQuantTest : public testing::WithParamInterface<MHATuple>,
                         virtual public SubgraphBaseTest, public CPUTestsBase  {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MHATuple> &obj) {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        size_t patternType;
        std::string targetName;
        std::string expectedNode;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNode, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=(";
        for (const auto& shape : inputShapes) {
            results << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        results << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                results << CommonTestUtils::vec2str(item) << "_";
            }
        }
        for (size_t i = 0; i < inputPrecisions.size(); i++) {
            results << "InPRC" << std::to_string(i) << "=" << inputPrecisions[i] << "_";
        }
        for (size_t i = 0; i < matMulIn0Precisions.size(); i++) {
            results << "MatMulIn0PRC" << std::to_string(i) << "=" << matMulIn0Precisions[i] << "_";
        }
        results << "patternType=" << patternType;
        results << "expect=" << expectedNode;
        results << "targetDevice=" << targetName;

        return results.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_element_type().is_real())
                tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(funcInput.get_element_type(), targetInputStaticShapes[i], 0.0f, 1.5f);
            else
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 255, 0, 1);


            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        abs_threshold = 0.1f;

        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::vector<ElementType> matMulIn0Precisions;
        size_t patternType;
        std::string expectedNode;
        std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNode, targetDevice) = this->GetParam();

        init_input_shapes(inputShapes);

        if (patternType == 0) {
            function = initMHAQuantSubgraph0(inputDynamicShapes, inputPrecisions, matMulIn0Precisions);
        } else if (patternType == 1) {
            function = initMHAQuantSubgraph1(inputDynamicShapes, inputPrecisions, matMulIn0Precisions, true);
        } else if (patternType == 2) {
            function = initMHAQuantSubgraph1(inputDynamicShapes, inputPrecisions, matMulIn0Precisions, false);
        } else {
            FAIL() << "Unsupported MHA pattern type";
        }

        // Snippets MHA tokenization has limitations to avoid performance degradations. These limitations depend on target machine.
        // Just for testing, we disable these limitations to allow Snippets to tokenize pattern on all machines for validation.
        if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
            configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                                  InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
        }
    }
};

TEST_P(MHAQuantTest, CompareWithRefs) {
    std::vector<InputShape> inputShapes;
    std::vector<ElementType> inputPrecisions;
    std::vector<ElementType> matMulIn0Precisions;
    size_t patternType;
    std::string expectedNode;
    std::tie(inputShapes, inputPrecisions, matMulIn0Precisions, patternType, expectedNode, targetDevice) = this->GetParam();

    if (inputPrecisions[0] == ElementType::bf16 && !InferenceEngine::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    if (!InferenceEngine::with_cpu_x86_avx512_core_vnni())
        GTEST_SKIP();

    run();
    CheckNumberOfNodesWithType(compiledModel, expectedNode, 1);
}

namespace {

std::vector<std::vector<ngraph::Shape>> inputShapesQuant = {
    {{2, 7, 16, 9}, {2, 7, 16, 9}, {2, 1, 1, 7}, {2, 7, 16, 9}},
    {{2, 8, 16, 64}, {2, 8, 16, 64}, {2, 1, 1, 8}, {2, 8, 16, 64}},
    {{1, 384, 16, 64}, {1, 384, 16, 64}, {1, 1, 1, 384}, {1, 384, 16, 64}},
    {{2, 64, 16, 80}, {2, 64, 16, 80}, {2, 1, 1, 64}, {2, 64, 16, 80}},
    {{3, 96, 16, 64}, {3, 96, 16, 64}, {3, 1, 1, 96}, {3, 96, 16, 64}},
    {{2, 192, 16, 160}, {2, 192, 16, 160}, {2, 1, 1, 192}, {2, 192, 16, 160}},
    {{2, 4, 16, 8}, {2, 4, 16, 8}, {2, 1, 1, 4}, {2, 4, 16, 8}},
    {{1, 204, 13, 212},  {1, 204, 13, 212},  {1, 1, 1, 204}, {1, 204, 13, 212}},
    {{1, 207, 13, 211},  {1, 207, 13, 211},  {1, 1, 1, 207}, {1, 207, 13, 211}},
};

std::vector<std::vector<ElementType>> inputPrecisionsQuant = {
    { ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32 },
};

std::vector<std::vector<ElementType>> matMulIn0PrecisionsQuant = {
    { ElementType::i8, ElementType::i8 },
    { ElementType::i8, ElementType::u8 },
};

std::vector<size_t> patternTypesQuant = {
    0, 1, 2
};

INSTANTIATE_TEST_SUITE_P(smoke_MHAQuant, MHAQuantTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesQuant)),
                                ::testing::ValuesIn(inputPrecisionsQuant),
                                ::testing::ValuesIn(matMulIn0PrecisionsQuant),
                                ::testing::ValuesIn(patternTypesQuant),
                                ::testing::Values("MHA"),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MHAQuantTest::getTestCaseName);

} // namespace
} // namespace CPUSubgraphTestsDefinitions
