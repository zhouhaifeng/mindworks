// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Motivation:
// In a dynamic scenario, depending on the input shapes for the current node, we can either generate a new jit kernel or get an existing one from the cache.
// But the current single layer tests do not allow checking the case when the same kernel can be used for different nodes.
// This subgraph test contains 2 FQ nodes and allows us to check this case.

//  ------------------------------------    ------------------------------------
//  |             Input 0              |    |             Input 1              |
//  ------------------------------------    ------------------------------------
//                   |                                       |
//  ------------------------------------    ------------------------------------
//  |          FakeQuantize 0          |    |          FakeQuantize 1          |
//  ------------------------------------    ------------------------------------
//                   |                                       |
//                   |                      ------------------------------------
//                   |                      |Reshape (if !reshapeShape.empty())|
//                   |                      ------------------------------------
//                   |                                       |
//  ----------------------------------------------------------------------------
//  |                                 Concat                                   |
//  ----------------------------------------------------------------------------
//                                       |
//                                   --------
//                                   |Output|
//                                   --------

#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ngraph_functions/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace CPUTestUtils;
using namespace ov::test;
using namespace InferenceEngine;
using namespace ngraph;

namespace CPUSubgraphTestsDefinitions {

using InputShapesTuple = std::tuple<
        std::vector<InputShape>,                // fq dynamic data shapes
        std::vector<std::vector<SizeVector>>,   // fq range input shapes
        std::vector<int32_t>                    // reshape shape
>;

using FqSpecificParams = std::tuple<int64_t,                  // 'data' input low bounds
                                    int64_t,                  // 'data' input high bounds
                                    std::vector<float>,       // output low
                                    std::vector<float>,       // output high
                                    size_t>;                  // levels

typedef std::tuple<
        InputShapesTuple,                                   // fq input shapes and reshape shape
        FqSpecificParams,                                   // fq specific params
        std::pair<std::vector<float>, std::vector<float>>,  // il and ih values
        CPUSpecificParams,
        std::map<std::string, std::string>                  // Additional config (disable snippets or no)
> FakeQuantizeCacheTestParams;

class FakeQuantizeCacheTest : public testing::WithParamInterface<FakeQuantizeCacheTestParams>,
                         virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FakeQuantizeCacheTestParams> &obj) {
        InputShapesTuple inputShapesTuple;
        FqSpecificParams fqParams;
        std::pair<std::vector<float>, std::vector<float>> inputRangesValues;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapesTuple, fqParams, inputRangesValues, cpuParams, additionalConfig) = obj.param;

        std::vector<InputShape> shapes;
        std::vector<std::vector<SizeVector>> ranges;
        std::vector<int32_t> reshapeShape;
        std::tie(shapes, ranges, reshapeShape) = inputShapesTuple;

        int64_t inDataLowBounds, inDataHighBounds;
        std::vector<float> inputLow, inputHigh, outputLow, outputHigh;
        size_t levels;
        inputLow = inputRangesValues.first;
        inputHigh = inputRangesValues.second;
        std::tie(inDataLowBounds, inDataHighBounds, outputLow, outputHigh, levels) = fqParams;

        std::ostringstream results;

        for (size_t i = 0; i < shapes.size(); i++) {
            results << "FQ" << i << "_IS=(" << CommonTestUtils::partialShape2str({shapes[i].first}) << ")_";
            results << "TS=";
            for (const auto& shape : shapes[i].second) {
                results << "(" << CommonTestUtils::vec2str(shape) << ")_";
            }
            results << "RS=";
            for (const auto& range : ranges[i]) {
                results << "(" << CommonTestUtils::vec2str(range) << ")_";
            }
        }
        if (!reshapeShape.empty()) {
            results << "ReshapeShape=(" << CommonTestUtils::vec2str(reshapeShape) << ")_";
        }

        results << "LOW_BOUNDS=" << inDataLowBounds << "_";
        results << "HIGH_BOUNDS=" << inDataHighBounds << "_";
        results << "IL=" << CommonTestUtils::vec2str(inputLow) << "_";
        results << "IH=" << CommonTestUtils::vec2str(inputHigh) << "_";
        results << "OL=" << CommonTestUtils::vec2str(outputLow) << "_";
        results << "OH=" << CommonTestUtils::vec2str(outputHigh) << "_";
        results << "LEVELS=" << levels;

        results << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            results << "_PluginConf";
            for (auto& item : additionalConfig) {
                results << "_" << item.first << "=" << item.second;
            }
        }

        return results.str();
    }

protected:
    void SetUp() override {
        abs_threshold = 0.01f;

        InputShapesTuple inputShapesTuple;
        FqSpecificParams fqParams;
        std::pair<std::vector<float>, std::vector<float>> inputRangesValues;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapesTuple, fqParams, inputRangesValues,
                cpuParams, additionalConfig) = this->GetParam();

        std::vector<InputShape> shapesVec;
        std::vector<std::vector<SizeVector>> rangesVec;
        std::vector<int32_t> reshapeShape;
        std::tie(shapesVec, rangesVec, reshapeShape) = inputShapesTuple;

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        targetDevice = CommonTestUtils::DEVICE_CPU;

        init_input_shapes(shapesVec);

        size_t levels;
        std::vector<std::vector<float>> rangesBounds(RANGES_INPUT_NUMBER);
        rangesBounds[0] = inputRangesValues.first;
        rangesBounds[1] = inputRangesValues.second;
        std::tie(inDataLowBounds, inDataHighBounds, rangesBounds[2], rangesBounds[3], levels) = fqParams;

        ParameterVector ngraphParam;
        std::vector<std::shared_ptr<Node>> ngraphInputs;

        auto ngInPrec = element::Type_t::f32;

        for (size_t i = 0; i < inputDynamicShapes.size(); i++) {
            ngraphParam.push_back(std::make_shared<opset1::Parameter>(ngInPrec, inputDynamicShapes[i]));
            ngraphInputs.push_back(ngraphParam.back());
        }

        auto makeFQ = [&](int i) {
            auto extendData = [](const std::vector<float> &data, size_t newSize) {
                std::vector<float> extendedData(newSize);
                size_t oldSize = data.size();
                for (size_t i = 0; i < newSize; i++) {
                    extendedData[i] = data[i % oldSize];
                }
                return extendedData;
            };

            auto ranges = rangesVec[i];

            auto il = builder::makeConstant(ngInPrec, ranges[0], extendData(rangesBounds[0],
                std::accumulate(ranges[0].begin(), ranges[0].end(), 1, std::multiplies<size_t>())));
            auto ih = builder::makeConstant(ngInPrec, ranges[1], extendData(rangesBounds[1],
                std::accumulate(ranges[1].begin(), ranges[1].end(), 1, std::multiplies<size_t>())));
            auto ol = builder::makeConstant(ngInPrec, ranges[2], extendData(rangesBounds[2],
                std::accumulate(ranges[2].begin(), ranges[2].end(), 1, std::multiplies<size_t>())));
            auto oh = builder::makeConstant(ngInPrec, ranges[3], extendData(rangesBounds[3],
                std::accumulate(ranges[3].begin(), ranges[3].end(), 1, std::multiplies<size_t>())));

            auto fqNode = std::make_shared<opset5::FakeQuantize>(ngraphParam[i], il, ih, ol, oh, levels);
            fqNode->get_rt_info() = getCPUInfo();
            return fqNode;
        };

        std::shared_ptr<Node> lastNode0 = makeFQ(0);
        std::shared_ptr<Node> lastNode1 = makeFQ(1);

        if (!reshapeShape.empty()) {
            auto reshapeConstNode = builder::makeConstant(::element::Type(::element::Type_t::i32),
                                                                  {reshapeShape.size()}, reshapeShape);
            lastNode1 = std::make_shared<opset5::Reshape>(lastNode1, reshapeConstNode, false);
        }
        auto concat = builder::makeConcat({lastNode0, lastNode1}, 0);

        if (selectedType.empty()) {
           selectedType = getPrimitiveType() + "_FP32";
        }

        function = std::make_shared<Function>(concat, ngraphParam, "fq_cache");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                             targetInputStaticShapes[i],
                                                             inDataHighBounds - inDataLowBounds,
                                                             inDataLowBounds);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    const size_t RANGES_INPUT_NUMBER = 4;

    int64_t inDataLowBounds, inDataHighBounds;
};

TEST_P(FakeQuantizeCacheTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, "FakeQuantize");
}

namespace {

const std::vector<size_t> levels = {256};

int64_t dataLowBounds{-10}, dataHighBounds{10};

const std::vector<std::pair<std::vector<float>, std::vector<float>>> inputRanges = {
    {{0.0f, 1.0f}, {5.0f, 6.0f}},
};

const std::vector<float> outputLow{5.0f, 6.0f}, outputHigh{25.0f, 31.0f};

const auto specificParams = ::testing::Combine(::testing::Values(dataLowBounds),
                                               ::testing::Values(dataHighBounds),
                                               ::testing::Values(outputLow),
                                               ::testing::Values(outputHigh),
                                               ::testing::ValuesIn(levels));

const std::map<std::string, std::string> emptyConfig = {};
const std::map<std::string, std::string> disableSnippets = {
    {PluginConfigInternalParams::KEY_SNIPPETS_MODE, PluginConfigInternalParams::DISABLE}};


// 3D
std::vector<CPUSpecificParams> cpuParams_3D = {
    CPUSpecificParams({ncw}, {ncw}, {}, {}),
};

std::vector<InputShapesTuple> inputShapes_3D = {
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, -1, 43}, {{1, 10, 43}, {1, 20, 43}, {1, 10, 43}, {1, 20, 43}}},
            // input1
            {{-1, -1, 43}, {{1, 10, 43}, {1, 20, 43}, {1, 10, 43}, {1, 20, 43}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, // miss
            // input1
            {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, // hit
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 10, -1}, {{1, 10, 22}, {1, 10, 44}, {1, 10, 22}, {1, 10, 44}}},
            // input1
            {{-1, 10, -1}, {{1, 10, 22}, {1, 10, 44}, {1, 10, 22}, {1, 10, 44}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 10, 1}, {1, 10, 1}, {1, 10, 1}, {1, 10, 1}}, // miss
            // input1
            {{1, 10, 1}, {1, 10, 1}, {1, 10, 1}, {1, 10, 1}}, // hit
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 10, -1}, {{1, 10, 22}, {1, 10, 44}, {1, 10, 22}, {1, 10, 44}}},
            // input1
            {{-1, 10, -1}, {{1, 10, 22}, {1, 10, 44}, {1, 10, 22}, {1, 10, 44}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 10, 1}, {1, 10, 1}, {1, 10, 1}, {1, 10, 1}}, // miss
            // input1
            {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, // miss
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 10, -1}, {{1, 10, 22}, {1, 10, 22}}},
            // input1
            {{-1, 20, -1}, {{1, 20, 22}, {1, 20, 22}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 10, 1}, {1, 10, 1}, {1, 10, 1}, {1, 10, 1}}, // miss
            // input1
            {{1, 20, 1}, {1, 20, 1}, {1, 20, 1}, {1, 20, 1}}, // hit
        },
        // reshape shape
        {-1, 10, 22},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeCache_3D, FakeQuantizeCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_3D),
                                specificParams,
                                ::testing::ValuesIn(inputRanges),
                                ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_3D)),
                                ::testing::Values(disableSnippets)),
                        FakeQuantizeCacheTest::getTestCaseName);


// 4D
std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
};

std::vector<InputShapesTuple> inputShapes_4D = {
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, -1, -1, 43}, {{1, 17, 3, 43}, {1, 34, 3, 43}, {1, 17, 3, 43}, {1, 34, 3, 43}}},
            // input1
            {{-1, -1, -1, 43}, {{1, 17, 3, 43}, {1, 34, 3, 43}, {1, 17, 3, 43}, {1, 34, 3, 43}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, // miss
            // input1
            {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, // hit
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 47, -1, -1}, {{1, 47, 2, 22}, {1, 47, 3, 33}, {1, 47, 2, 22}, {1, 47, 3, 33}}},
            // input1
            {{-1, 47, -1, -1}, {{1, 47, 2, 22}, {1, 47, 3, 33}, {1, 47, 2, 22}, {1, 47, 3, 33}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 47, 1, 1}, {1, 47, 1, 1}, {1, 47, 1, 1}, {1, 47, 1, 1}}, // miss
            // input1
            {{1, 47, 1, 1}, {1, 47, 1, 1}, {1, 47, 1, 1}, {1, 47, 1, 1}}, // hit
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 47, -1, -1}, {{1, 47, 2, 22}, {1, 47, 3, 33}, {1, 47, 2, 22}, {1, 47, 3, 33}}},
            // input1
            {{-1, 47, -1, -1}, {{1, 47, 2, 22}, {1, 47, 3, 33}, {1, 47, 2, 22}, {1, 47, 3, 33}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 47, 1, 1}, {1, 47, 1, 1}, {1, 47, 1, 1}, {1, 47, 1, 1}}, // miss
            // input1
            {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, // miss
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 17, -1, -1}, {{1, 17, 2, 22}, {1, 17, 2, 22}}},
            // input1
            {{-1, 34, -1, -1}, {{1, 34, 2, 22}, {1, 34, 2, 22}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 17, 1, 1}, {1, 17, 1, 1}, {1, 17, 1, 1}, {1, 17, 1, 1}}, // miss
            // input1
            {{1, 34, 1, 1}, {1, 34, 1, 1}, {1, 34, 1, 1}, {1, 34, 1, 1}}, // hit
        },
        // reshape shape
        {-1, 17, 2, 22},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeCache_4D, FakeQuantizeCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_4D),
                                specificParams,
                                ::testing::ValuesIn(inputRanges),
                                ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
                                ::testing::Values(disableSnippets)),
                        FakeQuantizeCacheTest::getTestCaseName);


// 5D
std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
};

std::vector<InputShapesTuple> inputShapes_5D = {
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, -1, -1, -1, 43}, {{1, 17, 2, 3, 43}, {1, 34, 2, 3, 43}, {1, 17, 2, 3, 43}, {1, 34, 2, 3, 43}}},
            // input1
            {{-1, -1, -1, -1, 43}, {{1, 17, 2, 3, 43}, {1, 34, 2, 3, 43}, {1, 17, 2, 3, 43}, {1, 34, 2, 3, 43}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, // miss
            // input1
            {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, // hit
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 47, -1, -1, -1}, {{1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}, {1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}}},
            // input1
            {{-1, 47, -1, -1, -1}, {{1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}, {1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}}, // miss
            // input1
            {{1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}}, // hit
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 47, -1, -1, -1}, {{1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}, {1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}}},
            // input1
            {{-1, 47, -1, -1, -1}, {{1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}, {1, 47, 2, 3, 22}, {1, 47, 3, 2, 33}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}, {1, 47, 1, 1, 1}}, // miss
            // input1
            {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, // miss
        },
        // reshape shape
        {},
    },
    {
        // fq dynamic data shapes
        {
            // input0
            {{-1, 17, -1, -1, -1}, {{1, 17, 2, 3, 22}, {1, 17, 2, 3, 22}}},
            // input1
            {{-1, 34, -1, -1, -1}, {{1, 34, 2, 3, 22}, {1, 34, 2, 3, 22}}},
        },
        // fq range input shapes
        {
            // input0
            {{1, 17, 1, 1, 1}, {1, 17, 1, 1, 1}, {1, 17, 1, 1, 1}, {1, 17, 1, 1, 1}}, // miss
            // input1
            {{1, 34, 1, 1, 1}, {1, 34, 1, 1, 1}, {1, 34, 1, 1, 1}, {1, 34, 1, 1, 1}}, // hit
        },
        // reshape shape
        {-1, 17, 2, 3, 22},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeCache_5D, FakeQuantizeCacheTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_5D),
                                specificParams,
                                ::testing::ValuesIn(inputRanges),
                                ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
                                ::testing::Values(disableSnippets)),
                        FakeQuantizeCacheTest::getTestCaseName);

} // namespace
} // namespace CPUSubgraphTestsDefinitions
