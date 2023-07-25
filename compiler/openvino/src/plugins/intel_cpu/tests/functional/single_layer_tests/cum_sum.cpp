// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov;
using namespace test;

namespace CPULayerTestsDefinitions {

using cumSumParams = std::tuple<
    ngraph::element::Type, // data precision
    InputShape, // input shape
    std::int64_t, // axis
    bool, // exclusive
    bool>; // reverse

class CumSumLayerCPUTest : public testing::WithParamInterface<cumSumParams>,
                           public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<cumSumParams> obj) {
        ngraph::element::Type inputPrecision;
        InputShape shapes;
        std::int64_t axis;
        bool exclusive;
        bool reverse;
        std::tie(inputPrecision, shapes, axis, exclusive, reverse) = obj.param;

        std::ostringstream results;
        results << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
        results << "Prc=" << inputPrecision << "_";
        results << "Axis=" << axis << "_" << (exclusive ? "exclusive" : "") << "_" << (reverse ? "reverse" : "");
        return results.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        InputShape shapes;
        std::int64_t axis;
        bool exclusive;
        bool reverse;
        std::tie(inType, shapes, axis, exclusive, reverse) = this->GetParam();
        if (inType == ElementType::bf16)
            rel_threshold = 0.05f;

        selectedType = makeSelectedTypeStr("ref_any", inType);
        init_input_shapes({shapes});

        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto axisNode = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{}, std::vector<int64_t>{axis})->output(0);
        auto cumSum = ngraph::builder::makeCumSum(params[0], axisNode, exclusive, reverse);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ cumSum }, params, "CumSumLayerCPUTest");
        functionRefs = ngraph::clone_function(*function);
    }
};

TEST_P(CumSumLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "CumSum");
}

const ngraph::element::TypeVector inputPrecision = {
    ngraph::element::i8,
    ngraph::element::bf16,
    ngraph::element::f32
};

const std::vector<int64_t> axes = { 0, 1, 2, 3, 4, 5, 6 };
const std::vector<int64_t> negativeAxes = { -1, -2, -3, -4, -5, -6 };

const std::vector<bool> exclusive = { true, false };
const std::vector<bool> reverse = { true, false };

const std::vector<InputShape> inShapes = {
    {{-1},
     {{16}, {18}, {12}}},

    {{-1, -1},
     {{9, 15}, {18, 12}, {12, 12}}},

    {{-1, -1, -1},
     {{16, 10, 12}, {18, 12, 10}, {12, 18, 10}}},

    {{-1, -1, -1, -1},
     {{18, 20, 14, 12}, {19, 20, 14, 12}, {20, 22, 23, 25}}},

    {{-1, -1, -1, -1, -1},
     {{2, 4, 6, 2, 4}, {3, 5, 6, 3, 5}, {1, 4, 2, 6, 8}}},

    {{-1, -1, -1, -1, -1, -1},
     {{2, 4, 6, 2, 4, 2}, {3, 5, 6, 3, 5, 3}, {1, 4, 2, 6, 8, 1}}},

    {{{-1, -1, -1, -1, -1, -1, -1}},
     {{2, 4, 6, 2, 4, 2, 4}, {3, 5, 6, 3, 5, 3, 5}, {1, 4, 2, 6, 8, 1, 4}}},

    {{{2, 5}, {3, 7}, {4, 8}, {5, 7}, {2, 5}, {3, 7}, {1, 2}},
     {{2, 4, 6, 5, 4, 3, 1}, {3, 5, 6, 6, 5, 3, 1}, {5, 7, 4, 6, 3, 7, 2}}},

     {{{2, 5}, -1, {4, 8}, -1, -1, {3, 7}, -1},
      {{2, 4, 6, 5, 4, 3, 1}, {3, 5, 6, 6, 5, 3, 1}, {5, 7, 4, 6, 3, 7, 2}}}
};

const auto testCasesAxis_0 = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(inShapes),
    ::testing::Values(axes[0]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_1 = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 1, inShapes.end())),
    ::testing::Values(axes[1]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_2 = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 2, inShapes.end())),
    ::testing::Values(axes[2]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_3 = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 3, inShapes.end())),
    ::testing::Values(axes[3]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_4 = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 4, inShapes.end())),
    ::testing::Values(axes[4]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_5 = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 5, inShapes.end())),
    ::testing::Values(axes[5]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_6 = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 6, inShapes.end())),
    ::testing::Values(axes[6]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

const auto testCasesAxis_negative = ::testing::Combine(
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 6, inShapes.end())),
    ::testing::ValuesIn(negativeAxes),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse)
);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_0, CumSumLayerCPUTest, testCasesAxis_0, CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_1, CumSumLayerCPUTest, testCasesAxis_1, CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_2, CumSumLayerCPUTest, testCasesAxis_2, CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_3, CumSumLayerCPUTest, testCasesAxis_3, CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_4, CumSumLayerCPUTest, testCasesAxis_4, CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_5, CumSumLayerCPUTest, testCasesAxis_5, CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_6, CumSumLayerCPUTest, testCasesAxis_6, CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_negative_axes, CumSumLayerCPUTest, testCasesAxis_negative, CumSumLayerCPUTest::getTestCaseName);

} // namespace CPULayerTestsDefinitions
