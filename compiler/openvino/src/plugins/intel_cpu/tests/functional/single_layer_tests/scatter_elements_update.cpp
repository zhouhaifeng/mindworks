// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
using ScatterElementsUpdateShapes = std::vector<InputShape>;
using IndicesValues = std::vector<std::int64_t>;

struct ScatterElementsUpdateLayerParams {
    ScatterElementsUpdateShapes inputShapes;
    IndicesValues indicesValues;
};

using scatterUpdateParams = std::tuple<
    ScatterElementsUpdateLayerParams,
    std::int64_t,       // axis
    ElementType,        // input precision
    ElementType>;       // indices precision

class ScatterElementsUpdateLayerCPUTest : public testing::WithParamInterface<scatterUpdateParams>, public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<scatterUpdateParams> obj) {
        ScatterElementsUpdateLayerParams scatterParams;
        std::int64_t axis;
        ElementType inputPrecision;
        ElementType idxPrecision;
        std::tie(scatterParams, axis, inputPrecision, idxPrecision) = obj.param;
        const auto inputShapes = scatterParams.inputShapes;
        const auto indicesVals = scatterParams.indicesValues;

        std::ostringstream result;
        result << inputPrecision << "_IS=";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            for (const auto& targetShape : shape.second) {
                result << CommonTestUtils::vec2str(targetShape) << "_";
            }
            result << ")_";
        }
        result << "_indices_values=" << CommonTestUtils::vec2str(indicesVals)
               << "axis=" << axis << "_idx_precision=" << idxPrecision;
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto& inputPrecision = funcInput.get_element_type();
            const auto& targetShape = targetInputStaticShapes[i];
            ov::Tensor tensor;
            if (i == 1) {
                tensor = ov::Tensor{ inputPrecision, targetShape };
                const auto indicesVals = std::get<0>(this->GetParam()).indicesValues;
                if (inputPrecision == ElementType::i32) {
                    auto data = tensor.data<std::int32_t>();
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        data[i] = static_cast<std::int32_t>(indicesVals[i]);
                    }
                } else if (inputPrecision == ElementType::i64) {
                    auto data = tensor.data<std::int64_t>();
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        data[i] = indicesVals[i];
                    }
                } else {
                    IE_THROW() << "GatherElementsUpdate. Unsupported indices precision: " << inputPrecision;
                }
            } else {
                if (inputPrecision.is_real()) {
                    tensor = ov::test::utils::create_and_fill_tensor(inputPrecision, targetShape, 10, 0, 1000);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(inputPrecision, targetShape);
                }
            }
            inputs.insert({ funcInput.get_node_shared_ptr(), tensor });
        }
    }

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        ScatterElementsUpdateLayerParams scatterParams;
        std::int64_t axis;
        ElementType inputPrecision;
        ElementType idxPrecision;
        std::tie(scatterParams, axis, inputPrecision, idxPrecision) = this->GetParam();
        const auto inputShapes = scatterParams.inputShapes;
        const auto indicesDescr = scatterParams.indicesValues;

        init_input_shapes(inputShapes);
        selectedType = makeSelectedTypeStr("unknown", inputPrecision);

        auto dataParams = ngraph::builder::makeDynamicParams(inputPrecision, { inputDynamicShapes[0], inputDynamicShapes[2] });
        auto indicesParam = ngraph::builder::makeDynamicParams(idxPrecision, { inputDynamicShapes[1] });
        dataParams[0]->set_friendly_name("Param_1");
        indicesParam[0]->set_friendly_name("Param_2");
        dataParams[1]->set_friendly_name("Param_3");

        auto axisNode = ngraph::opset3::Constant::create(idxPrecision, {}, { axis });
        auto scatter = std::make_shared<ngraph::opset3::ScatterElementsUpdate>(dataParams[0], indicesParam[0], dataParams[1], axisNode);

        ngraph::ParameterVector allParams{ dataParams[0], indicesParam[0], dataParams[1] };
        function = makeNgraphFunction(inputPrecision, allParams, scatter, "ScatterElementsUpdateLayerCPUTest");
    }
};

TEST_P(ScatterElementsUpdateLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ScatterUpdate");
}

const std::vector<std::int64_t> axes = { -3, -2, -1, 0, 1, 2 };

const std::vector<ScatterElementsUpdateLayerParams> scatterParams = {
    ScatterElementsUpdateLayerParams{
        ScatterElementsUpdateShapes{
            {{-1, -1, -1}, {{10, 12, 15}, {8, 9, 10}, {11, 8, 12}}},
            {{-1, -1, -1}, {{1, 2, 4}, {2, 1, 4}, {4, 1, 2}}},
            {{-1, -1, -1}, {{1, 2, 4}, {2, 1, 4}, {4, 1, 2}}}
        },
        IndicesValues{1, 0, 4, 6, 2, 3, 7, 5},
    },
    ScatterElementsUpdateLayerParams{
        ScatterElementsUpdateShapes{
            {{-1, -1, -1, -1}, {{10, 9, 8, 12}, {8, 12, 10, 9}, {11, 10, 12, 9}}},
            {{-1, -1, -1, -1}, {{1, 2, 2, 2}, {1, 2, 1, 4}, {1, 2, 2, 2}}},
            {{-1, -1, -1, -1}, {{1, 2, 2, 2}, {1, 2, 1, 4}, {1, 2, 2, 2}}}
        },
        IndicesValues{1, 0, 4, 6, 2, 3, 7, 5},
    },
    ScatterElementsUpdateLayerParams{
        ScatterElementsUpdateShapes{
            {{{7, 15}, {9, 12}, {1, 12}, {8, 12}}, {{10, 9, 8, 12}, {8, 12, 10, 9}, {11, 10, 12, 9}}},
            {{-1, -1, -1, -1}, {{1, 2, 2, 2}, {1, 2, 1, 4}, {1, 2, 2, 2}}},
            {{-1, -1, -1, -1}, {{1, 2, 2, 2}, {1, 2, 1, 4}, {1, 2, 2, 2}}}
        },
        IndicesValues{1, 0, 4, 6, 2, 3, 7, 5},
    },
};

const std::vector<ElementType> inputPrecisions = {
    ElementType::f32,
    ElementType::i32,
};

const std::vector<ElementType> constantPrecisions = {
    ElementType::i32,
    ElementType::i64,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ScatterElementsUpdateLayerCPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterParams),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterElementsUpdateLayerCPUTest::getTestCaseName);
} // namespace CPULayerTestsDefinitions
