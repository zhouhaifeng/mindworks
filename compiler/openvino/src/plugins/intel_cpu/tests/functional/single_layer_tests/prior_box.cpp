// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <sstream>
#include <vector>

#include <openvino/core/partial_shape.hpp>
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using priorBoxSpecificParams =  std::tuple<
        std::vector<float>, // min_size
        std::vector<float>, // max_size
        std::vector<float>, // aspect_ratio
        std::vector<float>, // density
        std::vector<float>, // fixed_ratio
        std::vector<float>, // fixed_size
        bool,               // clip
        bool,               // flip
        float,              // step
        float,              // offset
        std::vector<float>, // variance
        bool>;              // scale_all_sizes

typedef std::tuple<
        priorBoxSpecificParams,
        ov::test::ElementType,        // net precision
        ov::test::ElementType,        // Input precision
        ov::test::ElementType,        // Output precision
        InferenceEngine::Layout,      // Input layout
        InferenceEngine::Layout,      // Output layout
        ov::test::InputShape,         // input shape
        ov::test::InputShape,         // image shape
        std::string> priorBoxLayerParams;

class PriorBoxLayerCPUTest : public testing::WithParamInterface<priorBoxLayerParams>,
        virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxLayerParams>& obj) {
        ov::test::ElementType netPrecision;
        ov::test::ElementType inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        ov::test::InputShape inputShapes;
        ov::test::InputShape imageShapes;
        std::string targetDevice;
        priorBoxSpecificParams specParams;
        std::tie(specParams,
                 netPrecision,
                 inPrc, outPrc, inLayout, outLayout,
                 inputShapes,
                 imageShapes,
                 targetDevice) = obj.param;

        ngraph::op::PriorBoxAttrs attributes;
        std::tie(
            attributes.min_size,
            attributes.max_size,
            attributes.aspect_ratio,
            attributes.density,
            attributes.fixed_ratio,
            attributes.fixed_size,
            attributes.clip,
            attributes.flip,
            attributes.step,
            attributes.offset,
            attributes.variance,
            attributes.scale_all_sizes) = specParams;

        std::ostringstream result;
        const char separator = '_';
        result << "IS="      << inputShapes << separator;
        result << "imageS="  << imageShapes << separator;
        result << "netPRC="  << netPrecision   << separator;
        result << "inPRC="   << inPrc << separator;
        result << "outPRC="  << outPrc << separator;
        result << "inL="     << inLayout << separator;
        result << "outL="    << outLayout << separator;
        result << "min_size=" << CommonTestUtils::vec2str(attributes.min_size) << separator;
        result << "max_size=" << CommonTestUtils::vec2str(attributes.max_size)<< separator;
        result << "aspect_ratio=" << CommonTestUtils::vec2str(attributes.aspect_ratio)<< separator;
        result << "density=" << CommonTestUtils::vec2str(attributes.density)<< separator;
        result << "fixed_ratio=" << CommonTestUtils::vec2str(attributes.fixed_ratio)<< separator;
        result << "fixed_size=" << CommonTestUtils::vec2str(attributes.fixed_size)<< separator;
        result << "variance=" << CommonTestUtils::vec2str(attributes.variance)<< separator;
        result << "step=" << attributes.step << separator;
        result << "offset=" << attributes.offset << separator;
        result << "clip=" << attributes.clip << separator;
        result << "flip=" << attributes.flip<< separator;
        result << "scale_all_sizes=" << attributes.scale_all_sizes << separator;
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        priorBoxSpecificParams specParams;

        InferenceEngine::Layout inLayout;
        InferenceEngine::Layout outLayout;
        ov::test::ElementType netPrecision;
        ov::test::ElementType inPrc;
        ov::test::ElementType outPrc;
        ov::test::InputShape inputShapes;
        ov::test::InputShape imageShapes;
        std::tie(specParams, netPrecision,
                 inPrc, outPrc, inLayout, outLayout,
                 inputShapes, imageShapes, targetDevice) = GetParam();

        selectedType = makeSelectedTypeStr("ref_any", ov::test::ElementType::i32);
        targetDevice = CommonTestUtils::DEVICE_CPU;

        init_input_shapes({ inputShapes, imageShapes });

        ngraph::op::PriorBoxAttrs attributes;
        std::tie(
            attributes.min_size,
            attributes.max_size,
            attributes.aspect_ratio,
            attributes.density,
            attributes.fixed_ratio,
            attributes.fixed_size,
            attributes.clip,
            attributes.flip,
            attributes.step,
            attributes.offset,
            attributes.variance,
            attributes.scale_all_sizes) = specParams;

        auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);

        auto shape_of_1 = std::make_shared<ngraph::opset3::ShapeOf>(params[0]);
        auto shape_of_2 = std::make_shared<ngraph::opset3::ShapeOf>(params[1]);
        auto priorBox = std::make_shared<ngraph::op::PriorBox>(
                shape_of_1,
                shape_of_2,
                attributes);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBox)};
        function = std::make_shared <ngraph::Function>(results, params, "priorBox");
    }
};

TEST_P(PriorBoxLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "PriorBox");
}

namespace {
const std::vector<ov::test::ElementType> netPrecisions = {
    ov::test::ElementType::i32,
    ov::test::ElementType::u64};

const std::vector<std::vector<float>> min_sizes = {{256.0f}};

const std::vector<std::vector<float>> max_sizes = {{315.0f}};

const std::vector<std::vector<float>> aspect_ratios = {{2.0f}};

const std::vector<std::vector<float>> densities = {{1.0f}};

const std::vector<std::vector<float>> fixed_ratios = {{}};

const std::vector<std::vector<float>> fixed_sizes = {{}};

const std::vector<bool> clips = {false, true};

const std::vector<bool> flips = {false, true};

const std::vector<float> steps = {1.0f};

const std::vector<float> offsets = {0.0f};

const std::vector<std::vector<float>> variances = {{}};

const std::vector<bool> scale_all_sizes = { false, true};

const std::vector<ov::test::InputShape> inputShape = {
    {{300, 300}, {{300, 300}}},
    {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{300, 300}, {150, 150}}},
    {{{150, 300}, {150, 300}}, {{300, 300}, {150, 150}}}
};

const std::vector<ov::test::InputShape> imageShape = {
    {{32, 32}, {{32, 32}}},
    {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{32, 32}, {16, 16}}},
    {{{16, 32}, {16, 32}}, {{32, 32}, {16, 16}}}
};

const auto layerSpecificParams = ::testing::Combine(
    ::testing::ValuesIn(min_sizes),
    ::testing::ValuesIn(max_sizes),
    ::testing::ValuesIn(aspect_ratios),
    ::testing::ValuesIn(densities),
    ::testing::ValuesIn(fixed_ratios),
    ::testing::ValuesIn(fixed_sizes),
    ::testing::ValuesIn(clips),
    ::testing::ValuesIn(flips),
    ::testing::ValuesIn(steps),
    ::testing::ValuesIn(offsets),
    ::testing::ValuesIn(variances),
    ::testing::ValuesIn(scale_all_sizes));

INSTANTIATE_TEST_SUITE_P(smoke_PriorBox, PriorBoxLayerCPUTest,
    ::testing::Combine(
        layerSpecificParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::ElementType::undefined),
        ::testing::Values(ov::test::ElementType::undefined),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(imageShape),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    PriorBoxLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
