// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/loop.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace ov;
using namespace test;
using namespace ngraph::helpers;

namespace CPULayerTestsDefinitions {

enum LOOP_IN_TYPE {
    INVARIANT,
    MERGED
};

using LoopParams = typename std::tuple<
        InputLayerType,                                                    // TripCount is a constant?
        int64_t,                                                           // TripCount, -1 means infinity
        bool,                                                              // Execution condition
        std::vector<InputShape>,                                           // InputShapes
        std::vector<LOOP_IN_TYPE>,                                         // Type
        ElementType>;                                                      // Input element type


class LoopLayerCPUTest : public testing::WithParamInterface<LoopParams>,
                         virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LoopParams> obj) {
        InputLayerType trip_count_type;
        int64_t trip_count;
        bool exec_cond;
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        ElementType netType;
        std::tie(trip_count_type, trip_count, exec_cond, shapes, types, netType) = obj.param;

        std::ostringstream result;
        for (size_t i = 0; i < shapes.size(); i++) {
            result << "Input" << i << "_";
            result << "IS=" << CommonTestUtils::partialShape2str({shapes[i].first}) << "_";
            result << "TS=";
            for (const auto& item : shapes[i].second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << "types=";
        for (auto type : types)
            result << type << "_";
        result << "trip_count_type=" << trip_count_type << "_";
        result << "trip_count=" << trip_count << "_";
        result << "exec_cond=" << exec_cond << "_";
        result << "netType=" << netType;
        return result.str();
}

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        // trip count
        size_t i = 0;
        if (funcInputs[i].get_node_shared_ptr()->get_friendly_name() == "trip_count") {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                                 funcInput.get_shape(), 10, 1);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            i++;
        }

        // parameters for body
        for (; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                                 targetInputStaticShapes[i], 15, 0, 32768);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        InputLayerType trip_count_type;
        int64_t trip_count;
        bool exec_cond;
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        ElementType netType;
        std::tie(trip_count_type, trip_count, exec_cond, shapes, types, netType) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(shapes);

        auto params = ngraph::builder::makeDynamicParams(netType, inputDynamicShapes);

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(shapes.size(), ngraph::PartialShape::dynamic());
        ngraph::ParameterVector body_params;
        for (const auto &pshape : body_params_shapes) {
            body_params.emplace_back(std::make_shared<ngraph::opset1::Parameter>(netType, pshape));
        }

        auto body_condition_const = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);
        auto exec_condition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, exec_cond);
        std::shared_ptr<ngraph::Node> trip_count_input;
        int shift = 0;
        if (trip_count_type == InputLayerType::PARAMETER) {
            for (auto& target : targetStaticShapes)
                target.insert(target.begin(), ngraph::Shape{});
            trip_count_input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{1});
            trip_count_input->set_friendly_name("trip_count");
            params.insert(params.begin(), ov::as_type_ptr<ngraph::opset5::Parameter>(trip_count_input));
            shift++;
        } else {
            trip_count_input = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, trip_count);
        }

        // Body
        std::shared_ptr<ngraph::Node> Zo = body_params[0];
        for (size_t i = 1; i < body_params.size(); ++i) {
            Zo = std::make_shared<ngraph::op::v1::Add>(body_params[i], Zo);
        }

        auto body = std::make_shared<ov::Model>(ngraph::OutputVector{body_condition_const, Zo},
                                                       body_params);

        auto loop = std::make_shared<ngraph::opset5::Loop>(trip_count_input, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        for (size_t i = 0; i < body_params.size(); ++i) {
            if (types[i] == LOOP_IN_TYPE::INVARIANT) {
                loop->set_invariant_input(body_params[i], params[shift + i]);
            } else if (types[i] == LOOP_IN_TYPE::MERGED) {
                // todo: support several merged inputs
                // now supported only one in this sample
                loop->set_merged_input(body_params[i], params[shift + i], Zo);
            }
        }

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition_const, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

        auto result0 = std::make_shared<ngraph::opset5::Result>(out0);
        auto result1 = std::make_shared<ngraph::opset5::Result>(out1);
        auto result2 = std::make_shared<ngraph::opset5::Result>(out2);
        function = std::make_shared<ov::Model>(ngraph::ResultVector{result0, result1, result2}, params, "loop");
    }
};

class LoopWhileLayerCPUTest : public LoopLayerCPUTest {
protected:
    // body:
    // while (i < 10)
    //  x += 2
    //  i += 2

    void SetUp() override {
        InputLayerType trip_count_type;
        int64_t trip_count;
        bool exec_cond;
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        std::tie(trip_count_type, trip_count, exec_cond, shapes, types, inType) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(shapes);
        for (auto& target : targetStaticShapes)
            target.insert(target.begin(), ngraph::Shape{});

        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);

        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(shapes.size(), ngraph::PartialShape::dynamic());
        ngraph::ParameterVector body_params = { std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, ngraph::Shape{}) };
        for (const auto &pshape : body_params_shapes) {
            body_params.emplace_back(std::make_shared<ngraph::opset1::Parameter>(inType, pshape));
        }

        auto exec_condition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, exec_cond);
        auto trip_count_input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, ngraph::Shape{});
        trip_count_input->set_friendly_name("trip_count");
        params.insert(params.begin(), trip_count_input);

        // Body
        auto const_body_cond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{}, 10);
        auto const_body_step = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{}, 2);
        auto less = std::make_shared<ngraph::opset5::Less>(body_params[0], const_body_cond);
        auto exec_idx = std::make_shared<ngraph::opset5::Add>(body_params[0], const_body_step);

        auto node_const = std::make_shared<ngraph::opset5::Constant>(inType, ngraph::Shape{}, 2);
        auto node = std::make_shared<ngraph::opset5::Add>(body_params[1], node_const);

        // reference ngraph function is resized by input static shapes in tests but
        // loop with pad in body has different input shape in each infer request so tests don't support it.
        // Alternative - eltwise instead of pad
        // const std::vector<int64_t> begin(inputDynamicShapes[0].rank().get_length(), 1);
        // const std::vector<int64_t> end(inputDynamicShapes[0].rank().get_length(), 0);
        // auto node = ngraph::builder::makePad(body_params[1], begin, end, .0f, PadMode::CONSTANT);

        auto body = std::make_shared<ov::Model>(ngraph::OutputVector{less, exec_idx, node}, body_params);

        auto loop = std::make_shared<ngraph::opset5::Loop>(params[0], exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        loop->set_merged_input(body_params[0], params[0], exec_idx);
        loop->set_merged_input(body_params[1], params[1], node);

        auto out0 = loop->get_iter_value(exec_idx, -1);
        auto out1 = loop->get_iter_value(node, -1);

        auto result0 = std::make_shared<ngraph::opset5::Result>(out0);
        auto result1 = std::make_shared<ngraph::opset5::Result>(out1);
        function = std::make_shared<ov::Model>(ngraph::ResultVector{ result0, result1 }, params, "loop");
    }
};

class LoopForDiffShapesLayerCPUTest : public LoopLayerCPUTest {
    // parameter                   back edge
    //    |                 |-------------------|
    //    |-----------------|                   |
    //  StridedSlice       Add ----- Constant   |
    //    |                 |                   |
    //    |                 |-------------------|
    // ConcatOutput       Output

protected:
    void SetUp() override {
        InputLayerType trip_count_type;
        int64_t trip_count;
        bool exec_cond;
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        std::tie(trip_count_type, trip_count, exec_cond, shapes, types, inType) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(shapes);

        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(shapes.size(), ngraph::PartialShape::dynamic());
        ngraph::ParameterVector body_params;
        for (const auto &pshape : body_params_shapes) {
            body_params.emplace_back(std::make_shared<ngraph::opset1::Parameter>(inType, pshape));
        }

        auto body_condition_const = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);
        auto exec_condition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, exec_cond);
        std::shared_ptr<ngraph::Node> trip_count_input;
        int shift = 0;
        if (trip_count_type == InputLayerType::PARAMETER) {
            for (auto& target : targetStaticShapes)
                target.insert(target.begin(), ngraph::Shape{});
            trip_count_input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{1});
            trip_count_input->set_friendly_name("trip_count");
            params.insert(params.begin(), ov::as_type_ptr<ngraph::opset5::Parameter>(trip_count_input));
            shift++;
        } else {
            trip_count_input = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, trip_count);
        }

        // Body
        const auto axis = 1;
        auto s = ngraph::builder::makeSlice(body_params[0], {0}, {1}, {1}, {axis}, inType);
        auto constant = ngraph::builder::makeConstant(inType, std::vector<size_t>{1}, std::vector<float>{0.5});
        auto eltwise = std::make_shared<ov::op::v1::Add>(body_params[0], constant);

        auto body = std::make_shared<ov::Model>(ngraph::OutputVector{body_condition_const, s, eltwise}, body_params);

        auto loop = std::make_shared<ngraph::opset5::Loop>(trip_count_input, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        loop->set_merged_input(body_params[0], params[shift], eltwise);

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition_const, -1);
        auto out1 = loop->get_iter_value(eltwise, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(s, 0, 1, 1, -1, 1);

        auto result0 = std::make_shared<ngraph::opset5::Result>(out0);
        auto result1 = std::make_shared<ngraph::opset5::Result>(out1);
        auto result2 = std::make_shared<ngraph::opset5::Result>(out2);
        function = std::make_shared<ov::Model>(ngraph::ResultVector{result0, result1, result2}, params, "loop");
    }
};

class LoopForConcatLayerCPUTest : public LoopLayerCPUTest {
    // for 10:
    //   x = y + 10;
    //   y = concat(y, x)
    //   return y

protected:
    void SetUp() override {
        InputLayerType trip_count_type;
        int64_t trip_count;
        bool exec_cond;
        std::vector<InputShape> shapes;
        std::vector<LOOP_IN_TYPE> types;
        std::tie(trip_count_type, trip_count, exec_cond, shapes, types, inType) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(shapes);

        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);

        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(shapes.size(), ngraph::PartialShape::dynamic());
        auto body_params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);

        auto body_condition_const = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);
        auto exec_condition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, exec_cond);
        std::shared_ptr<ngraph::Node> trip_count_input;
        int shift = 0;
        if (trip_count_type == InputLayerType::PARAMETER) {
            for (auto& target : targetStaticShapes)
                target.insert(target.begin(), ngraph::Shape{});
            trip_count_input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{1});
            trip_count_input->set_friendly_name("trip_count");
            params.insert(params.begin(), ov::as_type_ptr<ngraph::opset5::Parameter>(trip_count_input));
            shift++;
        } else {
            trip_count_input = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, trip_count);
        }

        // Body
        auto constant = ngraph::builder::makeConstant(inType, std::vector<size_t>{1}, std::vector<float>{10});
        auto add = std::make_shared<ngraph::opset5::Add>(body_params[0], constant);
        auto concat = ngraph::builder::makeConcat({body_params[1], add}, 0);

        auto body = std::make_shared<ov::Model>(ngraph::OutputVector{body_condition_const, concat}, body_params);

        auto loop = std::make_shared<ngraph::opset5::Loop>(trip_count_input, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        loop->set_invariant_input(body_params[0], params[shift]);
        loop->set_merged_input(body_params[1], params[shift + 1], concat);

        auto out0 = loop->get_iter_value(body_condition_const, -1);
        auto out1 = loop->get_iter_value(concat, -1);

        auto result0 = std::make_shared<ngraph::opset5::Result>(out0);
        auto result1 = std::make_shared<ngraph::opset5::Result>(out1);
        function = std::make_shared<ov::Model>(ngraph::ResultVector{result0, result1}, params, "loop");
    }
};

TEST_P(LoopLayerCPUTest, CompareWithRefs) {
    run();
}

TEST_P(LoopWhileLayerCPUTest, CompareWithRefs) {
    run();
}

TEST_P(LoopForDiffShapesLayerCPUTest, CompareWithRefs) {
    run();
}

TEST_P(LoopForConcatLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ElementType> inputPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i8
};

std::vector<InputLayerType> trip_count_type { InputLayerType::CONSTANT, InputLayerType::PARAMETER };
std::vector<int64_t> trip_count { 0, 1, 5 };
std::vector<bool> exec_cond { true, false };

// dim[axis] = 1 because loop supports concatenation only with stride = part_size = 1
// the first loop suit test is with output concatenation
std::vector<std::vector<InputShape>> inputs = {
    {  //first test suit
        {   //dynamic shape for first input
            {-1, 1, -1},
            { // target static shapes
                {10, 1, 10},
                {1, 1, 1},
                {1, 1, 1},
                {5, 1, 3}
            }
        },
        {   //dynamic shape for second input
            {-1, -1, -1},
            { // target static shapes
                {1, 1, 1},
                {5, 1, 2},
                {5, 1, 2},
                {5, 1, 3}
            }
        },
        {   //dynamic shape for third input
            {-1, 1, -1},
            { // target static shapes
                {10, 1, 10},
                {5, 1, 2},
                {5, 1, 2},
                {5, 1, 3}
            }
        }
    },

    {  //second test suit
        {   //dynamic shape for first input
            {{1, 10}, 1, {1, 10}},
            { // target static shapes
                {8, 1, 8},
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
            }
        },
        {   //dynamic shape for second input
            {{1, 8}, 1, {1, 8}},
            { // target static shapes
                {8, 1, 8},
                {1, 1, 1},
                {1, 1, 1},
                {5, 1, 3}
            }
        },
        {   //dynamic shape for third input
            {{1, 10}, -1, {1, 10}},
            { // target static shapes
                {8, 1, 8},
                {1, 1, 1},
                {1, 1, 1},
                {5, 1, 3}
            }
        }
    },
};
std::vector<LOOP_IN_TYPE> types = {
        LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::MERGED
};

INSTANTIATE_TEST_SUITE_P(smoke_LoopForCommon, LoopLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(trip_count_type),
                                 ::testing::ValuesIn(trip_count),
                                 ::testing::ValuesIn(exec_cond),
                                 ::testing::ValuesIn(inputs),
                                 ::testing::Values(types),
                                 ::testing::ValuesIn(inputPrecisions)),
                         LoopLayerCPUTest::getTestCaseName);

std::vector<std::vector<InputShape>> inputs_2 = {
    {  //first test suit
        {   //dynamic shape
            {-1, -1},
            { // target static shapes
                {10, 10},
                {1, 1},
                {1, 1},
                {5, 3}
            }
        },
    },

    {  //second test suit
        {   //dynamic shape
            {{1, 10}, {1, 10}},
            { // target static shapes
                {5, 2},
                {2, 5},
                {5, 5},
                {5, 5}
            }
        },
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LoopWhileCommon, LoopWhileLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Values(trip_count_type[0]),
                                 ::testing::Values(-1),
                                 ::testing::Values(true),
                                 ::testing::ValuesIn(inputs_2),
                                 ::testing::Values(std::vector<LOOP_IN_TYPE>{}),
                                 ::testing::ValuesIn(inputPrecisions)),
                         LoopWhileLayerCPUTest::getTestCaseName);

std::vector<std::vector<InputShape>> inputs_3 = {
        {  // first test suit
            {
                {-1, -1, -1},
                { // target static shapes
                     {10, 1, 10},
                     {1, 10, 1},
                     {1, 10, 1},
                     {2, 2, 2},
                }
            },
        },
        {  // second test suit
            {
                {{0, 10}, {0, 10}, {0, 10}},
                { // target static shapes
                     {10, 5, 10},
                     {1, 10, 1},
                     {1, 10, 1},
                     {2, 1, 2},
                }
            },
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_LoopForDiffShapesConcat, LoopForDiffShapesLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(trip_count_type),
                                 ::testing::ValuesIn(trip_count),
                                 ::testing::ValuesIn(exec_cond),
                                 ::testing::ValuesIn(inputs_3),
                                 ::testing::Values(std::vector<LOOP_IN_TYPE>{}),
                                 ::testing::ValuesIn(inputPrecisions)),
                         LoopLayerCPUTest::getTestCaseName);

std::vector<std::vector<InputShape>> inputs_4 = {
        {  // first test suit
            {  // first input
                {-1, 10, 10},
                { // target static shapes
                    {10, 10, 10},
                    {5, 10, 10},
                    {5, 10, 10},
                    {8, 10, 10},
                }
            },
            {  // second input
                {-1, 10, 10},
                { // target static shapes
                    {0, 10, 10},
                    {0, 10, 10},
                    {0, 10, 10},
                    {0, 10, 10},
                }
            },
        },
        {  // second test suit
            {  // first input
                {{0, 10}, 10, 10},
                { // target static shapes
                    {10, 10, 10},
                    {5, 10, 10},
                    {5, 10, 10},
                    {8, 10, 10},
                }
            },
            {  // second input
                {-1, 10, 10},
                { // target static shapes
                    {0, 10, 10},
                    {0, 10, 10},
                    {0, 10, 10},
                    {0, 10, 10},
                }
            },
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_LoopForConcat, LoopForConcatLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(trip_count_type),
                                 ::testing::ValuesIn(trip_count),
                                 ::testing::ValuesIn(exec_cond),
                                 ::testing::ValuesIn(inputs_4),
                                 ::testing::Values(std::vector<LOOP_IN_TYPE>{}),
                                 ::testing::ValuesIn(inputPrecisions)),
                         LoopLayerCPUTest::getTestCaseName);

}  // namespace
} // namespace CPULayerTestsDefinitions
