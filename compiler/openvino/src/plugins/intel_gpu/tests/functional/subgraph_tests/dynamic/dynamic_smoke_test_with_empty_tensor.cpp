// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/shape_of.hpp"
#include "shared_test_classes/single_layer/strided_slice.hpp"
#include <shared_test_classes/single_layer/eltwise.hpp>
#include "shared_test_classes/single_layer/gather.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>, // input shapes
        ElementType, // Network precision
        TargetDevice, // Device name
        std::map<std::string, std::string> // Additional network configuration
> emptyTensorTestParamsSet;

const std::vector<ElementType> netPrecisions = {
    ElementType::i32,
};

class EmptyTensorDynamicGPUTest : public testing::WithParamInterface<emptyTensorTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<emptyTensorTestParamsSet>& obj) {
        emptyTensorTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> inputShapes;
        ElementType netType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << CommonTestUtils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "NetType=" << netType << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
     void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
          inputs.clear();
          const auto& funcInputs = function->inputs();
          for (size_t i = 0; i < funcInputs.size(); ++i) {
            auto node = funcInputs[i].get_node_shared_ptr();
            auto tensor = ov::runtime::Tensor(node->get_element_type(), targetInputStaticShapes[i]);
            if (i == 0) {
                // All zero inputs for non_zero op
                auto tensor_ptr = static_cast<int32_t*>(tensor.data());
                for (size_t j = 0; j < ov::shape_size(targetInputStaticShapes[i]); ++j) {
                    tensor_ptr[j] = 0;
                }
            } else {
                // Random inputs for concat
                tensor = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                 targetInputStaticShapes[i],
                                                                 80,
                                                                 0,
                                                                 8);
            }
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
          }
     }

     void SetUp() override {
          emptyTensorTestParamsSet basicParamsSet = this->GetParam();
          std::vector<InputShape> inputShapes;
          ElementType netType;
          std::map<std::string, std::string> additionalConfig;
          std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;

          init_input_shapes(inputShapes);
          const auto AllZeroData = inputDynamicShapes[0];
          const auto ConcatInputData = inputDynamicShapes[1];
          auto params = builder::makeDynamicParams(netType, {AllZeroData, ConcatInputData});
          auto paramOuts =
              helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
          const ElementType intInputsPrecision = ElementType::i32;
          auto nonzeroEmptyResultOp = std::make_shared<ngraph::opset3::NonZero>(paramOuts[0]);

          auto convertEmptyInputOp = ngraph::builder::makeConversion(nonzeroEmptyResultOp,
                                                                     ElementType::i32,
                                                                     ngraph::helpers::ConversionTypes::CONVERT);
          auto concatPartialInputEmptyOp =
              ngraph::builder::makeConcat({convertEmptyInputOp, paramOuts[1], convertEmptyInputOp},
                                          1);  // partially empty input / non empty output
          auto concatEmptyInputEmptyOutputOp =
              ngraph::builder::makeConcat({convertEmptyInputOp, convertEmptyInputOp, convertEmptyInputOp},
                                          1);  // all empty input/ all empty output

          std::vector<int64_t> squeezeDims = {0};
          auto squeezeDimsConst =
              std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ngraph::Shape{1}, squeezeDims);

          auto squeezeEmptyInputOp = std::make_shared<ngraph::opset1::Squeeze>(nonzeroEmptyResultOp, squeezeDimsConst);

          auto axisNode = ngraph::builder::makeConstant<int64_t>(intInputsPrecision, ov::Shape({1}), {0});
          auto gatherEmptyIndicesOp =
              std::make_shared<ov::op::v7::Gather>(paramOuts[0], squeezeEmptyInputOp, axisNode, 0);
          auto shapeofEmptyInputOp = std::make_shared<ngraph::opset3::ShapeOf>(gatherEmptyIndicesOp, ElementType::i32);
          ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(shapeofEmptyInputOp),
                                          std::make_shared<ngraph::opset1::Result>(concatPartialInputEmptyOp),
                                          std::make_shared<ngraph::opset1::Result>(concatEmptyInputEmptyOutputOp)};
          function = std::make_shared<ngraph::Function>(results, params, "result");

          auto nonzero = std::make_shared<ngraph::opset3::NonZero>(paramOuts[0]);
     }
};


TEST_P(EmptyTensorDynamicGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
std::map<std::string, std::string> emptyAdditionalConfig;
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        // Input for NonZero
        {{ov::Dimension::dynamic()}, {{30}, {40}, {50}, {10}, {7}}},
        // Input for Concat
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 0}, {1, 8}, {1, 0}, {1, 3}, {1, 20}}}
    },
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(netPrecisions), // netprec
                                                   ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_empty_tensor, EmptyTensorDynamicGPUTest,
                         testParams_smoke, EmptyTensorDynamicGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
