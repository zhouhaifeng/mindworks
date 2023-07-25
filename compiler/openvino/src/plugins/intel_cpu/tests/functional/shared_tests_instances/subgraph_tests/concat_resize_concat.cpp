// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include <ngraph/shape.hpp>
#include <ngraph/node.hpp>

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        ngraph::NodeTypeInfo,                        // Node type
        int,                                         // channels count
        int                                          // batch count
> ConcResizeConcParams;

class ConcatResizeConcatTest : public testing::WithParamInterface<ConcResizeConcParams>,
                               public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcResizeConcParams> &obj) {
        ngraph::NodeTypeInfo resize_type;
        int channels_count;
        int batch_count;
        std::tie(resize_type, channels_count, batch_count) = obj.param;
        std::ostringstream result;
        result << resize_type.name << "_";
        result << "Batches=" << batch_count << "_";
        result << "Channels=" << channels_count << "_";
        result << obj.index;
        return result.str();
}
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        ngraph::NodeTypeInfo resize_type;
        int channels_count;
        int batch_count;
        std::tie(resize_type, channels_count, batch_count) =  this->GetParam();

        std::vector<int> dims1({batch_count, channels_count, 2, 2});
        std::vector<int> dims2({batch_count, channels_count, 3, 3});

        std::vector<size_t> shape1({size_t(dims1[0]), size_t(dims1[1]), size_t(dims1[2]), size_t(dims1[3])});
        std::vector<size_t> shape2({size_t(dims2[0]), size_t(dims2[1]), size_t(dims2[2]), size_t(dims2[3])});
        auto inputNode1 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape(shape1));
        auto inputNode2 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape(shape1));
        auto inputNode3 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape(shape2));
        // concat layer
        ngraph::OutputVector concatNodes1;
        concatNodes1.push_back(inputNode1);
        concatNodes1.push_back(inputNode2);
        std::shared_ptr<ngraph::Node> inputNode = std::make_shared<ngraph::opset3::Concat>(concatNodes1, 1);

        // preresize layer
        ngraph::opset4::Interpolate::InterpolateAttrs attrs;
        attrs.mode = ngraph::opset4::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = ngraph::opset4::Interpolate::ShapeCalcMode::SIZES;
        attrs.coordinate_transformation_mode = ngraph::opset4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        attrs.nearest_mode = ngraph::opset4::Interpolate::NearestMode::CEIL;
        std::vector<int64_t> shape = {3, 3 };

        std::vector<float> scales = {1.5, 1.5 };
        auto outputShape = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{2}, shape.data());
        auto scalesShape = std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, ngraph::Shape{2}, scales.data());
        auto axes = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{2}, std::vector<int64_t>{2, 3});
        std::shared_ptr<ngraph::Node> preresizeNode = std::make_shared<ngraph::opset4::Interpolate>(inputNode, outputShape, scalesShape, axes, attrs);

        // concat layer
        ngraph::OutputVector concatNodes2;
        concatNodes2.push_back(preresizeNode);
        concatNodes2.push_back(inputNode3);
        std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Concat>(concatNodes2, 1);

        // Run shape inference on the nodes
        ngraph::NodeVector nodes;
        nodes.push_back(inputNode1);
        nodes.push_back(inputNode2);
        nodes.push_back(inputNode3);
        nodes.push_back(inputNode);
        nodes.push_back(preresizeNode);
        nodes.push_back(outputNode);

        // Create graph
        ngraph::ParameterVector inputs;
        inputs.push_back(inputNode1);
        inputs.push_back(inputNode2);
        inputs.push_back(inputNode3);
        ngraph::ResultVector outputs;
        outputs.push_back(std::make_shared<ngraph::opset1::Result>(outputNode));
        function = std::make_shared<ngraph::Function>(outputs, inputs);
    }
};

TEST_P(ConcatResizeConcatTest, CompareWithRefs) {
    Run();
}

namespace {

    const std::vector<int> batch_count = { 1, 2 };

    const std::vector<int> channel_count = { 1, 2 };


INSTANTIATE_TEST_SUITE_P(smoke_ConcResizeConc,
                        ConcatResizeConcatTest, ::testing::Combine(
                           ::testing::Values(ngraph::opset4::Interpolate::get_type_info_static()),
                           ::testing::ValuesIn(channel_count),
                           ::testing::ValuesIn(batch_count)),
                        ConcatResizeConcatTest::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
