// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_gather.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "ts_test_case.hpp"
#include "ts_test_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset10;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing::utils;

namespace transpose_sinking {
namespace testing {
namespace gather {

class GatherFactory : public IFactory {
public:
    explicit GatherFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        if (parent_nodes.size() == 3) {
            return std::make_shared<Gather>(parent_nodes[0], parent_nodes[1], parent_nodes[2]);
        }
        OPENVINO_ASSERT(false, "Unexpected number of inputs to Gather operation.");
    }
};

FactoryPtr CreateGatherFactory(const std::string& type_name) {
    return std::make_shared<GatherFactory>(type_name);
}
// ----------------------------------------------------------------------------

#undef CREATE_GATHER_FACTORY
#define CREATE_GATHER_FACTORY(type_name) CreateGatherFactory(#type_name)
// ----------------------------------------------------------------------------

shared_ptr<ov::Model> create_model(size_t main_node_idx,
                                   const ModelDescription& model_desc,
                                   size_t num_ops,
                                   const OutputVector& inputs_to_main) {
    auto new_inputs = model_desc.preprocess_inputs_to_main.apply(inputs_to_main);
    auto main_node = create_main_node(new_inputs, num_ops, model_desc.main_op[main_node_idx]);
    auto outputs = model_desc.preprocess_outputs_of_main.apply(main_node->outputs());
    return make_shared<ov::Model>(outputs, filter_parameters(inputs_to_main));
}

auto wrapper = [](const TestCase& test_case) {
    OPENVINO_ASSERT(test_case.model.main_op.size() == test_case.model_ref.main_op.size(),
                    "The number of main op (testing op) creator have to be the same for the testing model and for"
                    "the reference model.");
    return ::testing::Combine(::testing::Range<size_t>(0, test_case.num_main_ops.size()),
                              ::testing::Range<size_t>(0, test_case.model.main_op.size()),
                              ::testing::Values(test_case));
};

struct GatherForwardArguments {
    OutputVector inputs_to_main;
    Output<Node> new_Gather_first_input;
    AxisVector new_transpose_order;
};

auto test_forward_gather = [](const GatherForwardArguments& test_arguments) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSGatherForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = test_arguments.inputs_to_main;

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_GATHER_FACTORY(Gather)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_transpose = [&test_arguments](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        auto order = make_shared<Constant>(i32,
                                           Shape{test_arguments.new_transpose_order.size()},
                                           test_arguments.new_transpose_order);
        new_out_vec[0] = make_shared<Transpose>(out_vec[0], order);
        return new_out_vec;
    };
    auto new_constant = [&](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = out_vec[1];
        new_out_vec[2] = test_arguments.new_Gather_first_input;
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{2}}};
    test_case.model_ref.main_op = {CREATE_GATHER_FACTORY(Gather)};
    test_case.model_ref.preprocess_outputs_of_main = {{new_transpose}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

vector<GatherForwardArguments> tests_arguments_fw{
    {{{parameter(f32, {3, 4, 5, 6}), constant<int>(i32, {2}, {0, 2}), constant<int>(i32, {1}, {2})}},
     constant<int>(i32, {1}, {1}),
     AxisVector{3, 2, 1, 0}},
    {{parameter(f32, {2, 4}), constant<int>(i32, {}, {0}), constant<int>(i32, {1}, {1})},
     constant<int>(i32, {1}, {0}),
     AxisVector{0}},
    {{parameter(f32, {2, 4}), constant<int>(i32, {1}, {0}), constant<int>(i32, {1}, {1})},
     constant<int>(i32, {1}, {0}),
     AxisVector{1, 0}},
    {{parameter(f32, {2, 3, 4}), constant<int>(i32, {2, 3}, {0, 1, 0, 1, 0, 1}), constant<int>(i32, {1}, {1})},
     constant<int>(i32, {1}, {1}),
     AxisVector{3, 1, 2, 0}}};

INSTANTIATE_TEST_SUITE_P(TSCommonGatherForward_0, TSTestFixture, test_forward_gather(tests_arguments_fw[0]));
INSTANTIATE_TEST_SUITE_P(TSCommonGatherForward_1, TSTestFixture, test_forward_gather(tests_arguments_fw[1]));
INSTANTIATE_TEST_SUITE_P(TSCommonGatherForward_2, TSTestFixture, test_forward_gather(tests_arguments_fw[2]));
INSTANTIATE_TEST_SUITE_P(TSCommonGatherForward_3, TSTestFixture, test_forward_gather(tests_arguments_fw[3]));

struct GatherBackwardArguments {
    OutputVector inputs_to_main;
    Output<Node> new_Gather_first_input;
};

auto test_backward_gather = [](const GatherBackwardArguments& test_arguments) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSGatherBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = test_arguments.inputs_to_main;

    // Test model description:
    test_case.model.main_op = {CREATE_GATHER_FACTORY(Gather)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [&](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = out_vec[1];
        new_out_vec[2] = test_arguments.new_Gather_first_input;
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, new_constant}, {{0}, {2}}};
    test_case.model_ref.main_op = {CREATE_GATHER_FACTORY(Gather)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

vector<GatherBackwardArguments> tests_arguments_bw{
    {{{parameter(f32, {3, 4, 5, 6}), constant<int>(i32, {2}, {0, 2}), constant<int>(i32, {1}, {2})}},
     constant<int>(i32, {1}, {1})}};

INSTANTIATE_TEST_SUITE_P(TSCommonGatherBackward_0, TSTestFixture, test_backward_gather(tests_arguments_bw[0]));

// In some cases shape of 2nd input to Gather op (indices) has `1` dims which can
// prevent TransposeSinking in backward direction.
// We can get around this case by wrapping Transpose op with Squeeze+Unsqueeze pair.
auto test_backward_gather_optimization = [](const GatherBackwardArguments& test_arguments) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSGatherBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = test_arguments.inputs_to_main;

    // Test model description:
    test_case.model.main_op = {CREATE_GATHER_FACTORY(Gather)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto update_gather_inputs = [&](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = make_shared<Squeeze>(out_vec[1]);
        new_out_vec[2] = test_arguments.new_Gather_first_input;
        return new_out_vec;
    };

    auto unsqueeze_for = [&](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        auto axis = constant<int>(i32, {1}, {0});
        return {make_shared<Unsqueeze>(out_vec[0], axis)};
    };

    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, update_gather_inputs}, {{0}, {1, 2}}};
    test_case.model_ref.main_op = {CREATE_GATHER_FACTORY(Gather)};
    test_case.model_ref.preprocess_outputs_of_main = {{unsqueeze_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

vector<GatherBackwardArguments> tests_arguments_bw_optimization{
    {{{parameter(f32, {257, 8}), constant<int>(i32, {1, 2}, {0}), constant<int>(i32, {1}, {0})}},
     constant<int>(i32, {1}, {1})}};

INSTANTIATE_TEST_SUITE_P(TSCommonGatherBackwardOptimization_0,
                         TSTestFixture,
                         test_backward_gather_optimization(tests_arguments_bw_optimization[0]));
}  // namespace gather
}  // namespace testing
}  // namespace transpose_sinking
