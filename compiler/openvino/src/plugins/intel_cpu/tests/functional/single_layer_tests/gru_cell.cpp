// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using GRUCellCpuSpecificParams = typename std::tuple<
        std::vector<InputShape>,           // Shapes
        bool,                              // Using decompose to sub-ops transformation
        std::vector<std::string>,          // Activations
        float,                             // Clip
        bool,                              // Linear before reset
        ElementType,                       // Network precision
        CPUSpecificParams,                 // CPU specific params
        std::map<std::string, std::string> // Additional config
>;

class GRUCellCPUTest : public testing::WithParamInterface<GRUCellCpuSpecificParams>,
                            virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GRUCellCpuSpecificParams> &obj) {
        std::vector<InputShape> inputShapes;
        bool decompose, linearBeforeReset;
        std::vector<std::string> activations;
        float clip = 0.f;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, decompose, activations, clip, linearBeforeReset, netPrecision, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << CommonTestUtils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1 ? "_" : "");
            }
            result << "}_";
        }
        result << "decompose=" << decompose << "_";
        result << "activations=" << CommonTestUtils::vec2str(activations)  << "_";
        result << "clip=" << clip << "_";
        result << "linear=" << linearBeforeReset << "_";
        result << "netPrec=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto &item : additionalConfig) {
                if (item.second == InferenceEngine::PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        bool decompose, linearBeforeReset;
        std::vector<std::string> activations;
        float clip = 0.f;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, decompose, activations, clip, linearBeforeReset, netPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = CommonTestUtils::DEVICE_CPU;

        init_input_shapes(inputShapes);

        const size_t hiddenSize = targetStaticShapes.front()[1][1];
        const size_t inputSize = targetStaticShapes.front()[0][1];

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netPrecision);
        }

        auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
        std::vector<ngraph::Shape> WRB = {{3 * hiddenSize, inputSize}, {3 * hiddenSize, hiddenSize}, {(linearBeforeReset ? 4 : 3) * hiddenSize}};
        auto gruCellOp = ngraph::builder::makeGRU(
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)), WRB, hiddenSize, activations, {}, {}, clip, linearBeforeReset);

        function = makeNgraphFunction(netPrecision, params, gruCellOp, "GRUCell");
    }
};

TEST_P(GRUCellCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RNNCell");
}

namespace {
/* CPU PARAMS */
std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
       {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}};

CPUSpecificParams cpuParams{{nc, nc}, {nc}, {"ref_any"}, "ref_any"};

std::vector<bool> shouldDecompose{false};
// oneDNN supports only sigmoid-tanh
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}};
// oneDNN supports only zero clip
std::vector<float> clip = {0.f};
std::vector<bool> linearBeforeReset = {true, false};
std::vector<ElementType> netPrecisions = { ElementType::f32 };

const std::vector<std::vector<ov::test::InputShape>> staticShapes = {
    { { {}, { {1, 1} } }, // Static shapes
      { {}, { {1, 1} } } },
    { { {}, { {1, 1} } }, // Static shapes
      { {}, { {1, 10} } } },
    { { {}, { {1, 30} } }, // Static shapes
      { {}, { {1, 10} } } },
    { { {}, { {1, 30} } }, // Static shapes
      { {}, { {1, 1} } } },
    { { {}, { {3, 1} } }, // Static shapes
      { {}, { {3, 1} } } },
    { { {}, { {5, 1} } }, // Static shapes
      { {}, { {5, 1} } } },
    { { {}, { {5, 30} } }, // Static shapes
      { {}, { {5, 10} } } }
};

INSTANTIATE_TEST_SUITE_P(smoke_static, GRUCellCPUTest,
                ::testing::Combine(::testing::ValuesIn(staticShapes),
                                   ::testing::ValuesIn(shouldDecompose),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(linearBeforeReset),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParams),
                                   ::testing::ValuesIn(additionalConfig)),
                GRUCellCPUTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicShapes = {
    { { { {-1}, 1 },                       // Dynamic shape 0
        { {1, 1}, {3, 1}, {5, 1} } },      // Target shapes
      { { {-1}, 1 },                       // Dynamic shape 1
        { {1, 1}, {3, 1}, {5, 1} } } },    // Target shapes
    { { { {1, 10}, 30 },                   // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30} } },   // Target shapes
      { { {1, 10}, 10 },                   // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10} } } }, // Target shapes
    { { { {1, 10}, {25, 35} },             // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30} } },   // Target shapes
      { { {1, 10}, -1 },                   // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10} } } }, // Target shapes
    { { { {1, 10}, {25, 35} },             // Dynamic shape 0
        { {2, 30}, {5, 30}, {8, 30}, {2, 30}, {5, 30}, {8, 30} } },   // Target shapes
      { { {1, 10}, -1 },                   // Dynamic shape 1
        { {2, 10}, {5, 10}, {8, 10}, {2, 10}, {5, 10}, {8, 10} } } }  // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, GRUCellCPUTest,
                ::testing::Combine(::testing::ValuesIn(dynamicShapes),
                                   ::testing::ValuesIn(shouldDecompose),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(linearBeforeReset),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParams),
                                   ::testing::ValuesIn(additionalConfig)),
                GRUCellCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
