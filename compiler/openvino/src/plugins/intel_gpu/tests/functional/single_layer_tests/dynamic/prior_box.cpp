// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/shape_of.hpp"
#include "shared_test_classes/single_layer/strided_slice.hpp"
#include "shared_test_classes/single_layer/prior_box.hpp"
#include "shared_test_classes/single_layer/prior_box_clustered.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ngraph_functions/builders.hpp"
#include <string>
#include <openvino/pass/constant_folding.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

using ElementType = ov::element::Type_t;

namespace GPULayerTestsDefinitions {
enum class priorbox_type {
    V0,
    V8,
    Clustered
};
typedef std::tuple<
        InputShape,
        InputShape,
        ElementType,                // Net precision
        priorbox_type
> PriorBoxLayerGPUTestParamsSet;
class PriorBoxLayerGPUTest : public testing::WithParamInterface<PriorBoxLayerGPUTestParamsSet>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PriorBoxLayerGPUTestParamsSet> obj) {
        InputShape input1Shape;
        InputShape input2Shape;
        ElementType netPrecision;
        priorbox_type priorboxType;
        std::tie(input1Shape, input1Shape, netPrecision, priorboxType) = obj.param;

        std::ostringstream result;
        switch (priorboxType) {
            case priorbox_type::Clustered:
                result << "PriorBoxClusteredTest_";
            case priorbox_type::V0:
                result << "PriorBoxV0Test_";
            case priorbox_type::V8:
            default:
                result << "PriorBoxV8Test_";
        }
        result << std::to_string(obj.index) << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "I1S=";
        result << CommonTestUtils::partialShape2str({input1Shape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : input1Shape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")";
        result << "I2S=";
        result << CommonTestUtils::partialShape2str({input2Shape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : input2Shape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")";
        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_GPU;

        auto netPrecision = ElementType::undefined;
        InputShape input1Shape;
        InputShape input2Shape;
        priorbox_type priorboxType;
        std::tie(input1Shape, input2Shape, netPrecision, priorboxType) = this->GetParam();

        init_input_shapes({input1Shape, input2Shape});

        inType = ov::element::Type(netPrecision);
        outType = ElementType::f32;

        auto beginInput = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
        auto endInput = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {4});
        auto strideInput = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});

        auto functionParams = builder::makeDynamicParams(inType, inputDynamicShapes);
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset3::Parameter>(functionParams));

        auto shapeOfOp1 = std::make_shared<opset3::ShapeOf>(paramOuts[0], element::i32);
        auto stridedSliceOp1 = ngraph::builder::makeStridedSlice(shapeOfOp1, beginInput, endInput, strideInput, element::i32,
                                                                {0}, {1}, {0}, {0}, {0});

        auto shapeOfOp2 = std::make_shared<opset3::ShapeOf>(paramOuts[1], element::i32);
        auto stridedSliceOp2 = ngraph::builder::makeStridedSlice(shapeOfOp2, beginInput, endInput, strideInput, element::i32,
                                                                {0}, {1}, {0}, {0}, {0});

        switch (priorboxType) {
            case priorbox_type::Clustered: {
                ngraph::op::v0::PriorBoxClustered::Attributes attributes_clustered;

                attributes_clustered.widths = {86, 13, 57, 39, 68, 34, 142, 50, 23};
                attributes_clustered.heights = {44, 10, 30, 19, 94, 32, 61, 53, 17};
                attributes_clustered.variances = {0.1, 0.1, 0.2, 0.2};
                attributes_clustered.step = 16;
                attributes_clustered.step_widths = 0;
                attributes_clustered.step_heights = 0;
                attributes_clustered.offset = 0.5;
                attributes_clustered.clip = false;

                auto priorBoxOp = std::make_shared<ngraph::op::v0::PriorBoxClustered>(stridedSliceOp1, stridedSliceOp2, attributes_clustered);

                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBoxOp)};
                function = std::make_shared <ngraph::Function>(results, functionParams, "PriorBoxV0Function");
                break;
            }
            case priorbox_type::V0: {
                ngraph::op::v0::PriorBox::Attributes attributes_v0;

                attributes_v0.min_size = {64};
                attributes_v0.max_size = {300};
                attributes_v0.aspect_ratio = {2};
                attributes_v0.variance = {0.1, 0.1, 0.2, 0.2};
                attributes_v0.step = 16;
                attributes_v0.offset = 0.5;
                attributes_v0.clip = false;
                attributes_v0.flip = true;
                attributes_v0.scale_all_sizes = true;

                auto priorBoxOp = std::make_shared<ngraph::op::v0::PriorBox>(stridedSliceOp1, stridedSliceOp2, attributes_v0);

                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBoxOp)};
                function = std::make_shared <ngraph::Function>(results, functionParams, "PriorBoxV0Function");
                break;
            }
            case priorbox_type::V8:
            default: {
                ngraph::op::v8::PriorBox::Attributes attributes_v8;

                attributes_v8.min_size = {64};
                attributes_v8.max_size = {300};
                attributes_v8.aspect_ratio = {2};
                attributes_v8.variance = {0.1, 0.1, 0.2, 0.2};
                attributes_v8.step = 16;
                attributes_v8.offset = 0.5;
                attributes_v8.clip = false;
                attributes_v8.flip = true;
                attributes_v8.scale_all_sizes = true;
                attributes_v8.min_max_aspect_ratios_order = true;

                auto priorBoxOp = std::make_shared<ngraph::op::v8::PriorBox>(stridedSliceOp1, stridedSliceOp2, attributes_v8);

                ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBoxOp)};
                function = std::make_shared <ngraph::Function>(results, functionParams, "PriorBoxV8Function");
            }
        }
        ngraph::op::v8::PriorBox::Attributes attributes;
        attributes.min_size = {64};
        attributes.max_size = {300};
        attributes.aspect_ratio = {2};
        attributes.variance = {0.1, 0.1, 0.2, 0.2};
        attributes.step = 16;
        attributes.offset = 0.5;
        attributes.clip = false;
        attributes.flip = true;
        attributes.scale_all_sizes = true;
        attributes.min_max_aspect_ratios_order = true;

        auto priorBoxOp = std::make_shared<ngraph::op::v8::PriorBox>(stridedSliceOp1, stridedSliceOp2, attributes);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBoxOp)};
        function = std::make_shared <ngraph::Function>(results, functionParams, "PriorBoxFunction");
    }
};

TEST_P(PriorBoxLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
};

const std::vector<priorbox_type> mode = {
        priorbox_type::V0,
        priorbox_type::V8,
        priorbox_type::Clustered
};

std::vector<ov::test::InputShape> inShapesDynamic = {
        {
            {1, 3, -1, -1},
            {
                { 1, 3, 30, 30 },
                { 1, 3, 20, 20 },
                { 1, 3, 40, 40 }
            }
        },
};
std::vector<ov::test::InputShape> imgShapesDynamic = {
        {
            {1, 3, -1, -1},
            {
                { 1, 3, 224, 224 },
                { 1, 3, 300, 300 },
                { 1, 3, 200, 200 }
            }
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_prior_box_dynamic,
    PriorBoxLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(imgShapesDynamic),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(mode)),
    PriorBoxLayerGPUTest::getTestCaseName);
} // namespace

} // namespace GPULayerTestsDefinitions
