// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"


/*This test runs the following subgraph:

    param1      param2      param3      param4
       |          |          |           |
       |          |          |           |
    Softmax     Softmax    Softmax     Softmax
       |          |          |           |
       |          |          |           |
    Reshape     Reshape    Reshape     Reshape
       |          |          |           |
       |          |          |           |
       \          /          \          /
        \        /            \        /
         \      /              \      /
          Concat                Concat
               |                |
               |                |
              Reshape           Reshape
                    |           |
                    \          /
                     \        /
                      \      /
                       Concat
                          |
                        Softmax
                          
                        Result
  
  The main purpose of this test is checking the code path when all the nodes except Softmax use "in-place" memory mode.
  Softmax is used as a model of an arbitrary subgraph preceding the pattern.
*/

using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

using VectorShapes = std::vector<InputShape>;

class ConcatReshapeConcatSubgraphTest : public testing::WithParamInterface<VectorShapes>,
                                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<VectorShapes> obj) {
        VectorShapes& inputShapes = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << CommonTestUtils::vec2str(itr);
                }
            }
            result << ")";
        }
        return result.str();
    }

    void SetUp() override {
        constexpr size_t number_of_params = 4ul;
        constexpr size_t softmax_axis = 1ul;
        constexpr int concat_axis = 0;
        targetDevice = CommonTestUtils::DEVICE_CPU;
        auto netPrc = ov::element::f32;
        auto& InputShapes = this->GetParam();
        ASSERT_EQ(InputShapes.size(), number_of_params) << "Unexpected number of input shapes";
        init_input_shapes(InputShapes);
        auto input_params = ngraph::builder::makeDynamicParams(netPrc, inputDynamicShapes);

        ov::NodeVector first_level_reshapes;

        for (size_t i = 0; i < number_of_params; ++i) {
            auto soft_max = std::make_shared<ngraph::opset1::Softmax>(input_params[i], softmax_axis);
            auto reshape_param = ngraph::builder::makeConstant<int>(ov::element::i32, {1}, {0});
            auto reshape = std::make_shared<ngraph::opset1::Unsqueeze>(soft_max, reshape_param);
            first_level_reshapes.push_back(reshape);
        }

        auto concat1 = std::make_shared<ngraph::opset1::Concat>(ov::NodeVector{first_level_reshapes[0], first_level_reshapes[1]}, concat_axis);
        auto concat2 = std::make_shared<ngraph::opset1::Concat>(ov::NodeVector{first_level_reshapes[2], first_level_reshapes[3]}, concat_axis);

        ov::NodeVector second_level_reshapes;
        ov::NodeVector first_level_concats = {concat1, concat2};

        for (size_t i = 0; i < number_of_params / 2; ++i) {
            auto reshape_param = ngraph::builder::makeConstant<int>(ov::element::i32, {1}, {0});
            auto reshape = std::make_shared<ngraph::opset1::Unsqueeze>(first_level_concats[i], reshape_param);
            second_level_reshapes.push_back(reshape);
        }

        auto concat3 = std::make_shared<ngraph::opset1::Concat>(second_level_reshapes, concat_axis);
        auto soft_max = std::make_shared<ngraph::opset1::Softmax>(concat3, softmax_axis);

        ngraph::ResultVector results;
        for (size_t i = 0; i < soft_max->get_output_size(); i++)
            results.push_back(std::make_shared<ngraph::opset1::Result>(soft_max->output(i)));

        function = std::make_shared<ngraph::Function>(results, input_params, "ConcatReshapeConcatPattern");
        ov::pass::Serialize serializer("ngraph.xml", "ngraph.bin");
        serializer.run_on_model(function);
    }
};

TEST_P(ConcatReshapeConcatSubgraphTest, CompareWithRefs) {
    run();
    ov::pass::Serialize serializer("exec_graph_dyn.xml", "exec_graph_dyn.bin");
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(compiledModel.get_runtime_model()));
}

namespace {

const std::vector<std::vector<InputShape>> inputShapes = {
    // {
    //     // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
    //     {{2, 64}, {{2, 64}}}, // input 0
    //     {{2, 64}, {{2, 64}}}, // input 1
    //     {{2, 64}, {{2, 64}}}, // input 2
    //     {{2, 64}, {{2, 64}}}  // input 3
    // },
    {
        // {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{2, -1}, {{2, 64}}}, // input 0
        {{2, -1}, {{2, 64}}}, // input 1
        {{2, -1}, {{2, 64}}}, // input 2
        {{2, -1}, {{2, 64}}}  // input 3
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Concat_Reshape_Concat, ConcatReshapeConcatSubgraphTest,
                        ::testing::ValuesIn(inputShapes),
                        ConcatReshapeConcatSubgraphTest::getTestCaseName);
} // namespace
} // namespace SubgraphTestsDefinitions