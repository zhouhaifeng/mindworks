// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/strided_slice.hpp>
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace LayerTestsDefinitions;
using namespace ov;
using namespace test;

namespace CPULayerTestsDefinitions {

struct StridedSliceParams {
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisAxisMask;
};

typedef std::tuple<
        InputShape,                         // Input shapes
        StridedSliceParams,
        ngraph::helpers::InputLayerType,    // Secondary input types
        ElementType,                        // Element type
        CPUSpecificParams> StridedSliceLayerCPUTestParamSet;

class StridedSliceLayerCPUTest : public testing::WithParamInterface<StridedSliceLayerCPUTestParamSet>,
                                 virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<StridedSliceLayerCPUTestParamSet> obj) {
        InputShape shapes;
        StridedSliceParams params;
        ngraph::helpers::InputLayerType secondaryInputType;
        ElementType dataType;
        CPUSpecificParams cpuParams;
        std::tie(shapes, params, secondaryInputType, dataType, cpuParams) = obj.param;

        std::ostringstream results;
        results << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
        results << "secondaryInputType=" << secondaryInputType << "_";
        results << "netPRC=" << dataType << "_";
        results << "begin=" << CommonTestUtils::vec2str(params.begin) << "_";
        results << "end=" << CommonTestUtils::vec2str(params.end) << "_";
        results << "stride=" << CommonTestUtils::vec2str(params.strides) << "_";
        results << "begin_m=" << CommonTestUtils::vec2str(params.beginMask) << "_";
        results << "end_m=" << CommonTestUtils::vec2str(params.endMask) << "_";
        results << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.newAxisMask)) << "_";
        results << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.shrinkAxisMask)) << "_";
        results << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.ellipsisAxisMask)) << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }

protected:
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        std::vector<void*> inputValues = {ssParams.begin.data(), ssParams.end.data(), ssParams.strides.data()};

        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 10, 1, 1);
            } else {
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i], inputValues[i-1]};
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        InputShape shapes;
        ngraph::helpers::InputLayerType secondaryInputType;
        CPUSpecificParams cpuParams;
        ov::element::Type dataType;
        std::tie(shapes, ssParams, secondaryInputType, dataType, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType = makeSelectedTypeStr("ref", dataType);
        targetDevice = CommonTestUtils::DEVICE_CPU;
        std::vector<InputShape> input_shapes = {shapes};

        init_input_shapes({input_shapes});
        for (auto& targetShapes : targetStaticShapes) {
            targetShapes.push_back({ssParams.begin.size()});
            targetShapes.push_back({ssParams.end.size()});
            targetShapes.push_back({ssParams.strides.size()});
        }

        auto params = ngraph::builder::makeDynamicParams(dataType, inputDynamicShapes);
        std::shared_ptr<ngraph::Node> ss;
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            ov::Shape inShape = {ssParams.begin.size()};

            auto beginNode = std::make_shared<ngraph::opset1::Parameter>(ov::element::i64, inShape);
            auto endNode = std::make_shared<ngraph::opset1::Parameter>(ov::element::i64, inShape);
            auto strideNode = std::make_shared<ngraph::opset1::Parameter>(ov::element::i64, inShape);

            params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(beginNode));
            params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(endNode));
            params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(strideNode));

            ss = ngraph::builder::makeStridedSlice(params[0], beginNode, endNode, strideNode, inType, ssParams.beginMask,
                                                   ssParams.endMask, ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);
        } else {
            ss = ngraph::builder::makeStridedSlice(params[0], ssParams.begin, ssParams.end, ssParams.strides, inType, ssParams.beginMask,
                                                   ssParams.endMask, ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);
        }
        function = makeNgraphFunction(inType, params, ss, "StridedSlice");
    }

    StridedSliceParams ssParams;
};

TEST_P(StridedSliceLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "StridedSlice");
}

namespace {

const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {nChw8c}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {ndhwc}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {nchw}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams {{ncdhw}, {ncdhw}, {}, {}};

const std::vector<ElementType> inputPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i8
};

const std::vector<ngraph::helpers::InputLayerType> inputLayerTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER
};

const std::vector<InputShape> inputShapesDynamic2D = {
        {{-1, -1},
         {{32, 20}, {16, 16}, {24, 16}}},

        {{-1, 16},
         {{16, 16}, {20, 16}, {32, 16}}},

        {{{16, 32}, {16, 32}},
         {{16, 32}, {32, 16}, {24, 24}}},
};

const std::vector<StridedSliceParams> paramsPlain2D = {
        StridedSliceParams{ { 2, 5 }, { 16, 8 }, { 1, 1 }, { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { -10, -11 }, { -2, -3 }, { 1, 1 }, { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { -16, -17 }, { -2, -3 }, { 1, 1 }, { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 2, 44 }, { 55, -2 }, { 2, 3 }, { 0, 1 }, { 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 2, -7 }, { 1, -2 }, { 2, 3 }, { 1, 0 }, { 1, 0 },  { },  { },  { } },
        StridedSliceParams{ { 2 }, { 22 }, { 2 }, { 0 }, { 0 },  { },  { },  { } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Static_2D, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation({{32, 20}})),
                                 ::testing::ValuesIn(paramsPlain2D),
                                 ::testing::ValuesIn(inputLayerTypes),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::Values(emptyCPUSpec)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Dynamic_2D, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic2D),
                             ::testing::ValuesIn(paramsPlain2D),
                             ::testing::ValuesIn(inputLayerTypes),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::Values(emptyCPUSpec)),
                         StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceParams> testCasesCommon4D = {
        StridedSliceParams{ { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0 }, { 1, 3, 20, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 1, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 20, 20 }, { 1, 5, 25, 26 }, { 1, 1, 1, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 20 }, { 1, 2, 30, 30 }, { 1, 1, 2, 1 }, { 0, 0, 0, 1 }, { 0, 1, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10 }, { 1, 3, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 1, 1 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 0, 10 }, { 1, 5, 32, 30 }, { 1, 1, 1, 1 }, { 0, 1, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10 }, { 1, 8, 32, 18 }, { 1, 2, 1, 2 },  { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10 }, { 0, 32, 18 }, { 1, 1, 1 }, { 1, 1, 0 }, { 1, 1, 0 },  { },  { },  { 1, 0, 0 } },
        StridedSliceParams{ { 0, 4, 10 }, { 1, 8, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 },  { },  { },  { 0, 0, 1 } }
};

const std::vector<Shape> inputShapesStatic4D = {
    { 1, 5, 32, 32 }, { 2, 5, 32, 48 }
};

const std::vector<InputShape> inputShapesDynamic4D = {
        {{-1, -1, -1, -1},
         {{ 1, 5, 32, 32 }, { 2, 5, 32, 32 }, { 1, 5, 64, 64 }, {0, 0, 0, 0}}},

        {{-1, 5, -1, -1},
         {{ 1, 5, 32, 32 }, { 2, 5, 32, 32 }, { 3, 5, 32, 36 }, {0, 5, 0, 0}}},

        {{{1, 5}, 5, {32, 64}, {32, 64}},
         {{ 2, 5, 32, 32 }, { 1, 5, 48, 32 }, { 5, 5, 32, 32 }}},
};

const std::vector<CPUSpecificParams> CPUParamsCommon4D = {
        cpuParams_nchw,
        cpuParams_nhwc,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Static_4D, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic4D)),
                                 ::testing::ValuesIn(testCasesCommon4D),
                                 ::testing::ValuesIn(inputLayerTypes),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsCommon4D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic4D),
                             ::testing::ValuesIn(testCasesCommon4D),
                             ::testing::ValuesIn(inputLayerTypes),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::ValuesIn(CPUParamsCommon4D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceParams> testCasesBlocked4DSubset1 = {
        StridedSliceParams{ { 0, 0, 0, 0 }, { 1, 32, 32, 32 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 16, 0 }, { 1, 32, 32, 32 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0 }, { 1, 32, 32, 16 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0 }, { 1, 16, 32, 32 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
};

const std::vector<StridedSliceParams> testCasesBlocked4DSubset2 = {
       StridedSliceParams{ { 0, 0, 5, 4 }, { 1, 16, 28, 27 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, 16, 0, 0 }, { 1, 32, 10, 10 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, 0, 20, 20 }, { 1, 32, 25, 25 }, { 1, 1, 1, 1 }, { 0, 1, 0, 0 }, { 0, 1, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, 16, 2, 10 }, { 1, 32, 32, 20 }, { 1, 1, 2, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, 16, 0, 0 }, { 2, 64, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, 32, 0, 0 }, { 2, 50, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, 0, 0, 0 }, { 2, 12, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, -16, 0, 10 }, { 2, 100, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, -32, 0, 0 }, { 2, -12, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
       StridedSliceParams{ { 0, 10 }, { 0, 20 }, { 1, 1 }, { 1, 0 }, { 1, 0 },  { },  { },  { 1, 0 } },
       StridedSliceParams{ { 0, 16, 0 }, { 2, 32, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 },  { },  { },  { 0, 0, 1 } },
};

const std::vector<Shape> inputShapesBlockedStatic4DSubset1 = {
        { 1, 32, 32, 32 }, { 1, 32, 32, 64 }
};

const std::vector<Shape> inputShapesBlockedStatic4DSubset2 = {
        { 1, 64, 32, 32 }, { 1, 64, 32, 64 }
};

const std::vector<InputShape> inputShapesBlockedDynamic4DSubset1 = {
        {{-1, 32, -1, -1},
         {{ 1, 32, 32, 32 }, { 2, 32, 32, 32 }, { 3, 32, 32, 48 }}},

        {{{1, 5}, 32, {32, 64}, {32, 64}},
         {{ 2, 32, 32, 32 }, { 1, 32, 48, 32 }, { 5, 32, 32, 48 }}},
};

const std::vector<InputShape> inputShapesBlockedDynamic4DSubset2 = {
        {{-1, 64, -1, -1},
         {{ 1, 64, 64, 32 }, { 2, 64, 32, 32 }, { 3, 64, 32, 48 }}},

         {{{1, 5}, 64, {32, 64}, {32, 64}},
          {{ 2, 64, 32, 32 }, { 1, 64, 48, 32 }, { 1, 64, 64, 64 }}},
};

const std::vector<CPUSpecificParams> CPUParamsBlocked4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
};

const std::vector<ngraph::helpers::InputLayerType> inputLayerTypesBlocked = {
        ngraph::helpers::InputLayerType::CONSTANT,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Static_4D_Subset1, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic4DSubset1)),
                                 ::testing::ValuesIn(testCasesBlocked4DSubset1),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked4D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D_Subset1, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesBlockedDynamic4DSubset1),
                                 ::testing::ValuesIn(testCasesBlocked4DSubset1),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked4D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Static_4D_Subset2, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic4DSubset2)),
                                 ::testing::ValuesIn(testCasesBlocked4DSubset2),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked4D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D_Subset2, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesBlockedDynamic4DSubset2),
                                 ::testing::ValuesIn(testCasesBlocked4DSubset2),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked4D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceParams> testCasesCommon5D = {
        StridedSliceParams{ { 0, 2, 0, 5, 4 }, { 1, 4, 5, 28, 27 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0, 0 }, { 1, 5, 20, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 10, 0, 0 }, { 1, 3, 20, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 20, 20 }, { 1, 5, 20, 30, 26 }, { 1, 1, 1, 2, 2 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0, 20 }, { 1, 2, 20, 30, 30 }, { 1, 1, 2, 1, 1 }, { 0, 0, 0, 0, 1 }, { 0, 1, 0, 1, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10, 0 }, { 1, 5, 10, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 0, 1, 1, 0 }, { 0, 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 0, 10, 0 }, { 1, 5, 20, 32, 32 }, { 1, 1, 1, 1, 1 }, { 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0, 0 }, { 1, 5, 10, 16, 16 }, { 1, 1, 2, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
};

const std::vector<Shape> inputShapesStatic5D = {
        { 1, 5, 20, 32, 32 }, { 2, 5, 32, 32, 32 }
};

const std::vector<InputShape> inputShapesDynamic5D = {
        {{-1, -1, -1, -1, -1},
         {{ 1, 5, 32, 32, 32 }, { 1, 5, 32, 32, 48 }, { 1, 5, 64, 64, 64 }, { 1, 10, 32, 32, 32 }, {0, 0, 0, 0, 0}}},

        {{-1, 5, -1, -1, -1},
         {{ 1, 5, 32, 32, 48 }, { 1, 5, 32, 48, 32 }, { 1, 5, 48, 32, 32 }, {0, 5, 0, 0, 0}}},

        {{{1, 5}, 5, {32, 64}, {32, 64}, {32, 64}},
         {{ 2, 5, 32, 32, 32 }, { 1, 5, 48, 32, 32 }, { 5, 5, 32, 32, 48 }}},
};

const std::vector<CPUSpecificParams> CPUParamsCommon5D = {
        cpuParams_ncdhw,
        cpuParams_ndhwc,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Static_5D, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic5D)),
                                 ::testing::ValuesIn(testCasesCommon5D),
                                 ::testing::ValuesIn(inputLayerTypes),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsCommon5D)),
                        StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesDynamic5D),
                                 ::testing::ValuesIn(testCasesCommon5D),
                                 ::testing::ValuesIn(inputLayerTypes),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsCommon5D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

const std::vector<StridedSliceParams> testCasesBlocked5DSubset1 = {
        StridedSliceParams{ { 0, 0, 0, 5, 4 }, { 1, 16, 5, 28, 27 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0, 0 }, { 1, 16, 20, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 20, 20 }, { 1, 16, 20, 30, 26 }, { 1, 1, 1, 2, 2 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0, 20 }, { 1, 16, 20, 30, 30 }, { 1, 1, 2, 1, 1 }, { 0, 0, 0, 0, 1 }, { 0, 1, 0, 1, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10, 0 }, { 1, 16, 10, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 0, 1, 1, 0 }, { 0, 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 10, 0 }, { 1, 8, 20, 32, 32 }, { 1, 1, 1, 1, 1 }, { 0, 1, 0, 0, 0 }, { 0, 1, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0, 0 }, { 1, 16, 10, 16, 16 }, { 1, 1, 2, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
};

const std::vector<StridedSliceParams> testCasesBlocked5DSubset2 = {
        StridedSliceParams{ { 0, 0, 0, 5, 4 }, { 1, 16, 5, 28, 27 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0, 0 }, { 1, 16, 20, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0, 0 }, { 1, 16, 20, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 20, 20 }, { 1, 16, 20, 30, 26 }, { 1, 1, 1, 2, 2 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0, 20 }, { 1, 16, 20, 30, 30 }, { 1, 1, 2, 1, 1 }, { 0, 0, 0, 0, 1 }, { 0, 1, 0, 1, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10, 0 }, { 1, 16, 10, 32, 20 }, { 1, 1, 1, 1, 1 }, { 0, 0, 1, 1, 0 }, { 0, 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 10, 0 }, { 1, 8, 20, 32, 32 }, { 1, 1, 1, 1, 1 }, { 0, 1, 0, 0, 0 }, { 0, 1, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0, 0 }, { 1, 16, 10, 16, 16 }, { 1, 1, 2, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0, 0 }, { 1, 25, 20, 10, 10 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 16, 0, 0, 0 }, { 1, 25, 20, 10, 10 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 16, 0, 0, 0 }, { 1, 64, 20, 10, 10 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },  { },  { },  { } },
};

const std::vector<Shape> inputShapesBlockedStatic5DSubset1 = {
        { 1, 16, 32, 32, 32 }, { 2, 16, 32, 32, 32 }, { 2, 32, 32, 32, 32 }
};

const std::vector<Shape> inputShapesBlockedStatic5DSubset2 = {
        { 1, 64, 32, 32, 32 }, { 2, 64, 32, 64, 32 }, { 2, 64, 32, 32, 32 }
};

const std::vector<InputShape> inputShapesBlockedDynamic5DSubset1 = {
        {{-1, 16, -1, -1, -1},
         {{ 1, 16, 32, 32, 32 }, { 2, 16, 32, 32, 32 }, { 2, 16, 32, 32, 32 }}},

        {{{1, 5}, 16, {16, 32}, {16, 32}, {16, 32}},
         {{ 1, 16, 32, 32, 32 }, { 2, 16, 32, 32, 32 }, { 1, 16, 20, 32, 32 }}},
};

const std::vector<InputShape> inputShapesBlockedDynamic5DSubset2 = {
        {{-1, 64, -1, -1, -1},
         {{ 1, 64, 64, 32, 32 }, { 2, 64, 32, 32, 32 }, { 3, 64, 32, 48, 32 }}},

        {{{1, 5}, 64, {16, 32}, {16, 32}, {16, 32}},
         {{ 1, 64, 32, 32, 32 }, { 2, 64, 32, 32, 32 }, { 1, 64, 20, 32, 32 }}},
};

const std::vector<CPUSpecificParams> CPUParamsBlocked5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Static_5D_Subset1, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic5DSubset1)),
                                 ::testing::ValuesIn(testCasesBlocked5DSubset1),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked5D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D_Subset1, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesBlockedDynamic5DSubset1),
                                 ::testing::ValuesIn(testCasesBlocked5DSubset1),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked5D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Static_5D_Subset2, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic4DSubset2)),
                                 ::testing::ValuesIn(testCasesBlocked4DSubset2),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked4D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D_Subset2, StridedSliceLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesBlockedDynamic5DSubset2),
                                 ::testing::ValuesIn(testCasesBlocked5DSubset2),
                                 ::testing::ValuesIn(inputLayerTypesBlocked),
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::ValuesIn(CPUParamsBlocked5D)),
                         StridedSliceLayerCPUTest::getTestCaseName);

/* Descriptors check */

class StridedSliceLayerDescriptorCPUTest : public StridedSliceLayerCPUTest {};

TEST_P(StridedSliceLayerDescriptorCPUTest, DescriptorsCheck) {
    ASSERT_THROW(compile_model(), ov::Exception);
}

const std::vector<StridedSliceParams> testCasesDescriptors = {
        StridedSliceParams{ { 0, -4, 0, 0 }, { 0, 2147483647, 0, 0 }, { 1, 1, 1, 1 }, { 1, 0, 1, 1 }, { 1, 0, 1, 1 }, { },  { },  { } },
        StridedSliceParams{ { 0, 5, 0, 0 }, { 1, 20, 28, 27 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0 }, { 1, 2147483647, 32, 32 }, { 1, 2, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 0 }, { 1, 2147483647, 32, 32 }, { 1, 2, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  {0, 1, 0, 0 },  { } },
        StridedSliceParams{ { 0, 0, 0, 0 }, { 1, 2147483647, 32, 32 }, { 1, 2, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  {0, 0, 1, 0 },  { },  { } },
};

const std::vector<InputShape> inputShapesDescriptors = {
        {{}, {{ 1, 16, 32, 32 }}},
        {{}, {{ 1, 17, 32, 32 }}},
        {{1, -1, 32, 32}, {{ 1, 16, 32, 32 }, { 1, 32, 32, 32 }}}
};

INSTANTIATE_TEST_SUITE_P(smoke_StridedSliceLayerDescriptorCPUTest, StridedSliceLayerDescriptorCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesDescriptors),
                                 ::testing::ValuesIn(testCasesDescriptors),
                                 ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                 ::testing::Values(ElementType::f32),
                                 ::testing::Values(cpuParams_nChw8c)),
                         StridedSliceLayerDescriptorCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
