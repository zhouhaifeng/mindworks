// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::opset3;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using BucketizeCPUParamsTuple = std::tuple<InputShape,   // Data shape
                                           InputShape,   // Buckets shape
                                           bool,         // Right edge of interval
                                           ElementType,  // Data input precision
                                           ElementType,  // Buckets input precision
                                           ElementType   // Output precision
                                           >;

class BucketizeLayerCPUTest : public testing::WithParamInterface<BucketizeCPUParamsTuple>,
                              virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BucketizeCPUParamsTuple>& obj) {
        InputShape dataShape;
        InputShape bucketsShape;
        bool with_right_bound;
        ElementType inDataPrc;
        ElementType inBucketsPrc;
        ElementType netPrc;

        std::tie(dataShape, bucketsShape, with_right_bound, inDataPrc, inBucketsPrc, netPrc) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::partialShape2str({dataShape.first}) << "_"
               << CommonTestUtils::partialShape2str({bucketsShape.first}) << "_";

        result << "TS=";
        for (const auto& item : dataShape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
        result << "BS=";
        for (const auto& item : bucketsShape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }

        result << "with_right_bound=" << with_right_bound;
        result << "inDataPrc=" << inDataPrc << "_";
        result << "inBucketsPrc=" << inBucketsPrc << "_";
        result << "netPrc=" << netPrc << "_";
        return result.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto data_size = shape_size(targetInputStaticShapes[0]);
        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                                 targetInputStaticShapes[0],
                                                                                 data_size * 5,
                                                                                 0,
                                                                                 10,
                                                                                 7235346);

        ov::Tensor tensorBucket =
            ov::test::utils::create_and_fill_tensor_unique_sequence(funcInputs[1].get_element_type(),
                                                                    targetInputStaticShapes[1],
                                                                    0,
                                                                    10,
                                                                    8234231);

        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
        inputs.insert({funcInputs[1].get_node_shared_ptr(), tensorBucket});
    }

protected:
    void SetUp() override {
        InputShape dataShape;
        InputShape bucketsShape;
        bool with_right_bound;
        ElementType inDataPrc;
        ElementType inBucketsPrc;
        ElementType netPrc;

        targetDevice = CommonTestUtils::DEVICE_CPU;
        std::tie(dataShape, bucketsShape, with_right_bound, inDataPrc, inBucketsPrc, netPrc) = this->GetParam();
        init_input_shapes({dataShape, bucketsShape});

        auto data = std::make_shared<ngraph::op::Parameter>(inDataPrc, inputDynamicShapes[0]);
        data->set_friendly_name("a_data");
        auto buckets = std::make_shared<ngraph::op::Parameter>(inBucketsPrc, inputDynamicShapes[1]);
        buckets->set_friendly_name("b_buckets");
        auto bucketize = std::make_shared<ngraph::op::v3::Bucketize>(data, buckets, netPrc, with_right_bound);
        function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(bucketize),
                                                      ngraph::ParameterVector{data, buckets},
                                                      "Bucketize");
    }
};

TEST_P(BucketizeLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::test::InputShape> dataShapesDynamic = {
    {{ngraph::Dimension(1, 10), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
     {{1, 20, 20}, {3, 16, 16}, {10, 16, 16}}},
    {{ngraph::Dimension(1, 10), 3, 50, 50}, {{1, 3, 50, 50}, {2, 3, 50, 50}, {10, 3, 50, 50}}}};

const std::vector<ov::test::InputShape> bucketsShapesDynamic = {{{ngraph::Dimension::dynamic()}, {{5}, {20}, {100}}}};

const std::vector<ov::test::ElementType> inPrc = {ov::element::f32, ov::element::i64, ov::element::i32};
const std::vector<ov::test::ElementType> outPrc = {ov::element::i64, ov::element::i32};

const auto test_Bucketize_right_edge_Dynamic = ::testing::Combine(::testing::ValuesIn(dataShapesDynamic),
                                                                  ::testing::ValuesIn(bucketsShapesDynamic),
                                                                  ::testing::Values(true),
                                                                  ::testing::ValuesIn(inPrc),
                                                                  ::testing::ValuesIn(inPrc),
                                                                  ::testing::ValuesIn(outPrc));

const auto test_Bucketize_left_edge_Dynamic = ::testing::Combine(::testing::ValuesIn(dataShapesDynamic),
                                                                 ::testing::ValuesIn(bucketsShapesDynamic),
                                                                 ::testing::Values(false),
                                                                 ::testing::ValuesIn(inPrc),
                                                                 ::testing::ValuesIn(inPrc),
                                                                 ::testing::ValuesIn(outPrc));

INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_right_Dynamic,
                         BucketizeLayerCPUTest,
                         test_Bucketize_right_edge_Dynamic,
                         BucketizeLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_left_Dynamic,
                         BucketizeLayerCPUTest,
                         test_Bucketize_left_edge_Dynamic,
                         BucketizeLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace CPULayerTestsDefinitions
