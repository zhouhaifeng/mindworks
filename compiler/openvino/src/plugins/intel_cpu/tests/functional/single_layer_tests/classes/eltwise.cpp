// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"
#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string EltwiseLayerCPUTest::getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj) {
        subgraph::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

        std::ostringstream result;
        result << subgraph::EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<subgraph::EltwiseTestParams>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
}

ov::Tensor EltwiseLayerCPUTest::generate_eltwise_input(const ov::element::Type& type, const ngraph::Shape& shape) {
        struct gen_params {
            uint32_t range;
            int32_t start_from;
            int32_t resolution;

            gen_params(uint32_t range = 10, int32_t start_from = 0, int32_t resolution = 1)
                : range(range), start_from(start_from), resolution(resolution) {}
        };

        gen_params params = gen_params();
        if (type.is_real()) {
            switch (eltwiseType) {
                case ngraph::helpers::EltwiseTypes::POWER:
                    params = gen_params(6, -3);
                case ngraph::helpers::EltwiseTypes::MOD:
                case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
                    params = gen_params(2, 2, 8);
                    break;
                case ngraph::helpers::EltwiseTypes::DIVIDE:
                    params = gen_params(2, 2, 8);
                    break;
                case ngraph::helpers::EltwiseTypes::ERF:
                    params = gen_params(6, -3);
                    break;
                default:
                    params = gen_params(80, 0, 8);
                    break;
            }
        } else {
            params = gen_params(INT32_MAX, INT32_MIN);
        }
        return ov::test::utils::create_and_fill_tensor(type, shape, params.range, params.start_from, params.resolution);
    }

void EltwiseLayerCPUTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            inputs.insert({funcInput.get_node_shared_ptr(), generate_eltwise_input(funcInput.get_element_type(), targetInputStaticShapes[i])});
        }
}

void EltwiseLayerCPUTest::SetUp() {
        subgraph::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();
        std::vector<InputShape> shapes;
        ElementType netType;
        ngraph::helpers::InputLayerType secondaryInputType;
        CommonTestUtils::OpType opType;
        Config additional_config;
        std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, inType, outType, targetDevice, configuration) = basicParamsSet;

        if (ElementType::bf16 == netType) {
            rel_threshold = 2e-2f;
        } else if (ElementType::i32 == netType) {
            abs_threshold = 0;
        }

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        selectedType = makeSelectedTypeStr(getPrimitiveType(), netType);
        #if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
            if (eltwiseType == POWER) {
                selectedType = std::regex_replace(selectedType, std::regex("acl"), "ref");
            }
        #endif

        shapes.resize(2);
        switch (opType) {
            case CommonTestUtils::OpType::SCALAR: {
                std::vector<ngraph::Shape> identityShapes(shapes[0].second.size(), {1});
                shapes[1] = {{}, identityShapes};
                break;
            }
            case CommonTestUtils::OpType::VECTOR:
                if (shapes[1].second.empty()) {
                    shapes[1] = shapes[0];
                }
                break;
            default:
                FAIL() << "Unsupported Secondary operation type";
        }

        init_input_shapes(shapes);

        configuration.insert(additional_config.begin(), additional_config.end());
        auto parameters = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.front()});
        std::shared_ptr<ngraph::Node> secondaryInput;
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            secondaryInput = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.back()}).front();
            parameters.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
        } else {
            auto pShape = inputDynamicShapes.back();
            ngraph::Shape shape;
            if (pShape.is_static()) {
                shape = pShape.get_shape();
            } else {
                ASSERT_TRUE(pShape.rank().is_static());
                shape = std::vector<size_t>(pShape.rank().get_length(), 1);
                for (size_t i = 0; i < pShape.size(); ++i) {
                    if (pShape[i].is_static()) {
                        shape[i] = pShape[i].get_length();
                    }
                }
            }
            if (netType == ElementType::i32) {
                auto data_tensor = generate_eltwise_input(ElementType::i32, shape);
                auto data_ptr = reinterpret_cast<int32_t*>(data_tensor.data());
                std::vector<int32_t> data(data_ptr, data_ptr + ngraph::shape_size(shape));
                secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
            } else {
                auto data_tensor = generate_eltwise_input(ElementType::f32, shape);
                auto data_ptr = reinterpret_cast<float*>(data_tensor.data());
                std::vector<float> data(data_ptr, data_ptr + ngraph::shape_size(shape));
                secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
            }
        }
        auto eltwise = ngraph::builder::makeEltwise(parameters[0], secondaryInput, eltwiseType);
        function = makeNgraphFunction(netType, parameters, eltwise, "Eltwise");
}

TEST_P(EltwiseLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

namespace Eltwise {

const ov::AnyMap& additional_config() {
        static const ov::AnyMap additional_config;
        return additional_config;
}

const std::vector<ElementType>& netType() {
        static const std::vector<ElementType> netType = {
                ElementType::f32};
        return netType;
}

const std::vector<CommonTestUtils::OpType>& opTypes() {
        static const std::vector<CommonTestUtils::OpType> opTypes = {
                CommonTestUtils::OpType::VECTOR,
        };
        return opTypes;
}

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesBinInp() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesBinInp = {
                ngraph::helpers::EltwiseTypes::ADD,
                ngraph::helpers::EltwiseTypes::MULTIPLY,
        #if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
                ngraph::helpers::EltwiseTypes::SUBTRACT,                // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::DIVIDE,                  // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::FLOOR_MOD,               // TODO: Fix CVS-111875
        #endif
                ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        };
        return eltwiseOpTypesBinInp;
}

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesDiffInp() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesDiffInp = { // Different number of input nodes depending on optimizations
                ngraph::helpers::EltwiseTypes::POWER,
                // ngraph::helpers::EltwiseTypes::MOD // Does not execute because of transformations
        };
        return eltwiseOpTypesDiffInp;
}

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesBinDyn() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesBinDyn = {
                ngraph::helpers::EltwiseTypes::ADD,
                ngraph::helpers::EltwiseTypes::MULTIPLY,
        #if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::SUBTRACT,
        #endif
                ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        };
        return eltwiseOpTypesBinDyn;
}

const std::vector<CPUSpecificParams>& cpuParams_4D() {
        static const std::vector<CPUSpecificParams> cpuParams_4D = {
                CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
                CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
        };
        return cpuParams_4D;
}

const std::vector<CPUSpecificParams>& cpuParams_5D() {
        static const std::vector<CPUSpecificParams> cpuParams_5D = {
                CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
                CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
        };
        return cpuParams_5D;
}

const std::vector<std::vector<ov::Shape>>& inShapes_4D() {
        static const std::vector<std::vector<ov::Shape>> inShapes_4D = {
                {{2, 4, 4, 1}},
                {{2, 17, 5, 4}},
                {{2, 17, 5, 4}, {1, 17, 1, 1}},
                {{2, 17, 5, 1}, {1, 17, 1, 4}},
        };
        return inShapes_4D;
}

const std::vector<std::vector<ov::Shape>>& inShapes_5D() {
        static const std::vector<std::vector<ov::Shape>> inShapes_5D = {
                {{2, 4, 3, 4, 1}},
                {{2, 17, 7, 5, 4}},
                {{2, 17, 6, 5, 4}, {1, 17, 6, 1, 1}},
                {{2, 17, 6, 5, 1}, {1, 17, 1, 1, 4}},
        };
        return inShapes_5D;
}

const std::vector<ngraph::helpers::EltwiseTypes>& eltwiseOpTypesI32() {
        static const std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesI32 = {
                ngraph::helpers::EltwiseTypes::ADD,
                ngraph::helpers::EltwiseTypes::MULTIPLY,
        #if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) // TODO: Fix CVS-105430
                ngraph::helpers::EltwiseTypes::SUBTRACT,
                ngraph::helpers::EltwiseTypes::DIVIDE,
        #endif
                ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        };
        return eltwiseOpTypesI32;
}

const std::vector<ngraph::helpers::InputLayerType>& secondaryInputTypes() {
        static const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
                ngraph::helpers::InputLayerType::CONSTANT,
                ngraph::helpers::InputLayerType::PARAMETER,
        };
        return secondaryInputTypes;
}

const std::vector<std::vector<ngraph::Shape>>& inShapes_4D_1D() {
        static const std::vector<std::vector<ngraph::Shape>> inShapes_4D_1D = {
                {{2, 17, 5, 4}, {4}},
                {{1, 3, 3, 3}, {3}},
        };
        return inShapes_4D_1D;
}

const std::vector<CPUSpecificParams> & cpuParams_4D_1D_Constant_mode() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_1D_Constant_mode = {
                CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
                CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
        };
        return cpuParams_4D_1D_Constant_mode;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_1D_Parameter_mode() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_1D_Parameter_mode = {
                CPUSpecificParams({nchw, x}, {nchw}, {}, {})
        };
        return cpuParams_4D_1D_Parameter_mode;
}

const std::vector<std::vector<ngraph::Shape>>& inShapes_5D_1D() {
        static const std::vector<std::vector<ngraph::Shape>> inShapes_5D_1D = {
                {{2, 17, 5, 4, 10}, {10}},
                {{1, 3, 3, 3, 3}, {3}},
        };
        return inShapes_5D_1D;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_1D_parameter() {
        static const std::vector<CPUSpecificParams> cpuParams_5D_1D_parameter = {
                CPUSpecificParams({ncdhw, x}, {ncdhw}, {}, {})
        };
        return cpuParams_5D_1D_parameter;
}

const std::vector<InputShape>& inShapes_4D_dyn_param() {
        static const std::vector<InputShape> inShapes_4D_dyn_param = {
        {
                // dynamic
                {-1, {2, 15}, -1, -1},
                // target
                {
                {3, 2, 1, 1},
                {1, 7, 5, 1},
                {3, 3, 4, 11},
                }
        },
        {
                // dynamic
                {-1, {2, 25}, -1, -1},
                // target
                {
                {1, 2, 5, 1},
                {3, 7, 1, 10},
                {3, 3, 4, 11}
                }
        }
        };
        return inShapes_4D_dyn_param;
}

const std::vector<InputShape>& inShapes_5D_dyn_param() {
        static const std::vector<InputShape> inShapes_5D_dyn_param = {
        {
                // dynamic
                {-1, {2, 15}, -1, -1, -1},
                // target
                {
                {3, 2, 1, 1, 1},
                {1, 7, 5, 1, 12},
                {3, 3, 4, 11, 6},
                }
        },
        {
                // dynamic
                {-1, {2, 25}, -1, -1, -1},
                // target
                {
                {1, 2, 5, 1, 5},
                {3, 7, 1, 10, 1},
                {3, 3, 4, 11, 6}
                }
        }
        };
        return inShapes_5D_dyn_param;
}

const std::vector<InputShape>& inShapes_5D_dyn_const() {
    static const std::vector<InputShape> inShapes_5D_dyn_const = {
    {
        // dynamic
        {3, 2, -1, -1, -1},
        // target
        {
            {3, 2, 1, 1, 1},
            {3, 2, 5, 1, 7},
            {3, 2, 1, 6, 1},
            {3, 2, 4, 11, 2},
        }
    },
};
    return inShapes_5D_dyn_const;
}

const std::vector<std::vector<InputShape>>& inShapes_4D_dyn_const() {
    static const std::vector<std::vector<InputShape>> inShapes_4D_dyn_const = {
        {
                {
                // dynamic
                {3, 2, -1, -1},
                // target
                {
                        {3, 2, 1, 1},
                        {3, 2, 5, 1},
                        {3, 2, 1, 6},
                        {3, 2, 4, 11},
                }
                }
        },
        {
                {
                // dynamic
                {{1, 10}, 2, 5, 6},
                // target
                {
                        {3, 2, 5, 6},
                        {1, 2, 5, 6},
                        {2, 2, 5, 6},
                }
                }
        },
    };
    return inShapes_4D_dyn_const;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_1D_constant() {
    static const std::vector<CPUSpecificParams> cpuParams_5D_1D_constant = {
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
    };
    return cpuParams_5D_1D_constant;
}

} // namespace Eltwise
} // namespace CPULayerTestsDefinitions
