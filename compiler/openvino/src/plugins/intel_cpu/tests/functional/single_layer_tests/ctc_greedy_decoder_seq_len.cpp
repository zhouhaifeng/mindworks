// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using CtcGreedyDecoderSeqLenParams = std::tuple<size_t,   // Batch size N
                                                size_t,   // Sequence length T
                                                size_t>;  // Number of classes

using InputShapeParams = std::pair<std::vector<ov::Dimension>,                  // bounds N, T, C, blank
                                   std::vector<CtcGreedyDecoderSeqLenParams>>;  // target input dimensions

using InputElementParams = std::vector<ElementType>;

using CTCGreedyDecoderSeqLenLayerCPUTestParams = std::tuple<InputShapeParams,    // Input Shape
                                                            InputElementParams,  // Input precision
                                                            ElementType,         // Index Type
                                                            bool                 // mergeRepeated
                                                            >;
inline ngraph::ParameterVector makeDynamicParams(const std::vector<ElementType>& types,
                                          const std::vector<ov::PartialShape>& shapes) {
    ngraph::ParameterVector outs;
    NGRAPH_CHECK(types.size() == shapes.size());
    for (size_t i = 0; i < types.size(); i++) {
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(types[i], shapes[i]);
        outs.push_back(paramNode);
    }
    return outs;
}

class CTCGreedyDecoderSeqLenLayerCPUTest : public testing::WithParamInterface<CTCGreedyDecoderSeqLenLayerCPUTestParams>,
                                           virtual public SubgraphBaseTest,
                                           public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CTCGreedyDecoderSeqLenLayerCPUTestParams>& obj) {
        InputElementParams inType;
        bool mergeRepeated;
        InputShapeParams shapes;
        ElementType indexType;
        std::tie(shapes, inType, indexType, mergeRepeated) = obj.param;
        std::ostringstream results;
        results << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& shape : shapes.second) {
            size_t N;
            size_t T;
            size_t C;
            std::tie(N, T, C) = shape;
            results << "{" << N << "," << T << "," << C << "}"
                    << "_";
        }
        for (const auto& type : inType) {
            results << "Prc=" << type << "_";
        }

        results << "IndexType=" << indexType << "_";

        results << "MergeRepeated=" << mergeRepeated;

        return results.str();
    }

protected:
    void SetUp() override {
        InputElementParams inType;
        bool mergeRepeated;
        InputShapeParams shapes;
        ElementType indexType;
        std::tie(shapes, inType, indexType, mergeRepeated) = GetParam();
        selectedType = "ref_any_FP32";
        targetDevice = CommonTestUtils::DEVICE_CPU;
        ASSERT_EQ(shapes.first.size(), 4);
        const auto& in_dyn_N = shapes.first[0];
        const auto& in_dyn_T = shapes.first[1];
        const auto& in_dyc_C = shapes.first[2];
        const auto& in_dyc_blank = shapes.first[3];
        const size_t blank_rank = in_dyc_blank.get_length();
        ASSERT_TRUE(blank_rank == 0 || blank_rank == 1);
        inputDynamicShapes = {ov::PartialShape{in_dyn_N, in_dyn_T, in_dyc_C},
                              ov::PartialShape{in_dyn_N},
                              blank_rank == 0 ? ov::PartialShape{} : ov::PartialShape{1}};

        for (auto& shape : shapes.second) {
            size_t N;
            size_t T;
            size_t C;
            std::tie(N, T, C) = shape;
            if (blank_rank == 0)
                targetStaticShapes.push_back({{N, T, C}, {N}, {}});
            else
                targetStaticShapes.push_back({{N, T, C}, {N}, {1}});
        }

        auto params = makeDynamicParams(inType, inputDynamicShapes);
        auto ctcGreedyDecoderSeqLen = std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(params[0],
                                                                                           params[1],
                                                                                           params[2],
                                                                                           mergeRepeated,
                                                                                           indexType,
                                                                                           indexType);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoderSeqLen)};
        function = std::make_shared<ngraph::Function>(results, params, "CTCGreedyDecoderSeqLenCPU");
    };

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto& dataShape = targetInputStaticShapes[0];
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0) {
                if (funcInput.get_element_type().is_real()) {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                     targetInputStaticShapes[i],
                                                                     10,
                                                                     0,
                                                                     1000);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                     targetInputStaticShapes[i]);
                }
            } else if (i == 1) {
                const auto seqLen = dataShape[1];
                const auto B = dataShape[0];
                std::mt19937 gen(42);
                std::uniform_int_distribution<unsigned long> dist(1, seqLen);

                std::vector<int32_t> sequenceLenData(B, 0);
                for (size_t b = 0; b < B; b++) {
                    const int len = dist(gen);
                    sequenceLenData[b] = len;
                }
                tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                if (funcInput.get_element_type() == ElementType::i32) {
                    auto begin = tensor.data<int32_t>();
                    std::copy(sequenceLenData.begin(), sequenceLenData.end(), begin);
                } else if (funcInput.get_element_type() == ElementType::i64) {
                    auto begin = tensor.data<int64_t>();
                    std::copy(sequenceLenData.begin(), sequenceLenData.end(), begin);
                }

            } else if (i == 2) {
                // blank should be valid class type
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                 targetInputStaticShapes[i],
                                                                 dataShape[2],
                                                                 0);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(CTCGreedyDecoderSeqLenLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "CTCGreedyDecoderSeqLen");
}

namespace {
// Common params
// ElementType::f16 is not support by CPU Plugin yet
const std::vector<ElementType> inputType = {ElementType::f32, ElementType::i32, ElementType::i32};
const std::vector<bool> mergeRepeated{true, false};
const std::vector<ElementType> indexType = {ElementType::i64, ElementType::i32};
const std::vector<InputShapeParams> inputShapesCTCDecoder = {
    {{ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{0}},
     {CtcGreedyDecoderSeqLenParams{1, 1, 1},
      CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
    {{ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{1}},
     {CtcGreedyDecoderSeqLenParams{1, 1, 1},
      CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
    {{ov::Dimension{1, 5}, ov::Dimension{1, 6}, ov::Dimension{1, 60}, ov::Dimension{0}},
     {CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
    {{ov::Dimension{1, 5}, ov::Dimension{1, 6}, ov::Dimension{1, 60}, ov::Dimension{1}},
     {CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inputShapesCTCDecoder),
                                           ::testing::Values(inputType),
                                           ::testing::ValuesIn(indexType),
                                           ::testing::ValuesIn(mergeRepeated));

INSTANTIATE_TEST_SUITE_P(smoke_CtcGreedyDecoderSeqLenCPU,
                         CTCGreedyDecoderSeqLenLayerCPUTest,
                         basicCases,
                         CTCGreedyDecoderSeqLenLayerCPUTest::getTestCaseName);
}  // namespace

}  // namespace CPULayerTestsDefinitions
