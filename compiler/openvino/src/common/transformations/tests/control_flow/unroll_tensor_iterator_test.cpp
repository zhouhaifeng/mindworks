// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, UnrollTensorIteratorGRUCell) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset4::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(gru_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(gru_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto axis_split = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split = std::make_shared<opset4::Split>(X, axis_split, 2);
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze_1 = std::make_shared<opset4::Squeeze>(split->output(0), axis);
        auto squeeze_2 = std::make_shared<opset4::Squeeze>(split->output(1), axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384}, b_val);

        auto gru_cell_1 = std::make_shared<opset4::GRUCell>(squeeze_1, Y, W, R, B, 128);
        auto gru_cell_2 = std::make_shared<opset4::GRUCell>(squeeze_2, gru_cell_1, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<opset4::Unsqueeze>(gru_cell_1, axis);
        auto unsqueeze_2 = std::make_shared<opset4::Unsqueeze>(gru_cell_2, axis);
        auto concat = std::make_shared<opset4::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(concat);
        // auto res_ti_2 = std::make_shared<opset4::Result>(unsqueeze_2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollTensorIteratorRNNCell) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset4::RNNCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(rnn_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto axis_split = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split = std::make_shared<opset4::Split>(X, axis_split, 2);
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze_1 = std::make_shared<opset4::Squeeze>(split->output(0), axis);
        auto squeeze_2 = std::make_shared<opset4::Squeeze>(split->output(1), axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128}, b_val);

        auto rnn_cell_1 = std::make_shared<opset4::RNNCell>(squeeze_1, Y, W, R, B, 128);
        auto rnn_cell_2 = std::make_shared<opset4::RNNCell>(squeeze_2, rnn_cell_1, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<opset4::Unsqueeze>(rnn_cell_1, axis);
        auto unsqueeze_2 = std::make_shared<opset4::Unsqueeze>(rnn_cell_2, axis);
        auto concat = std::make_shared<opset4::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(concat);
        // auto res_ti_2 = std::make_shared<opset4::Result>(unsqueeze_2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollTensorIteratorLSTMCell) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset4::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(lstm_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y, Z});

        ngraph::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto axis_split = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split = std::make_shared<opset4::Split>(X, axis_split, 2);
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze_1 = std::make_shared<opset4::Squeeze>(split->output(0), axis);
        auto squeeze_2 = std::make_shared<opset4::Squeeze>(split->output(1), axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell_1 = std::make_shared<opset4::LSTMCell>(squeeze_1, Y, Z, W, R, B, 128);
        auto lstm_cell_2 = std::make_shared<opset4::LSTMCell>(squeeze_2, lstm_cell_1, Z, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<opset4::Unsqueeze>(lstm_cell_1, axis);
        auto unsqueeze_2 = std::make_shared<opset4::Unsqueeze>(lstm_cell_2, axis);
        auto concat = std::make_shared<opset4::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(concat);
        // auto res_ti_2 = std::make_shared<opset4::Result>(unsqueeze_2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollTensorIteratorGRUCellSingleIteration) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset4::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(gru_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(gru_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze_1 = std::make_shared<opset4::Squeeze>(X, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384}, b_val);

        auto gru_cell_1 = std::make_shared<opset4::GRUCell>(squeeze_1, Y, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<opset4::Unsqueeze>(gru_cell_1, axis);

        auto res_ti_1 = std::make_shared<opset4::Result>(unsqueeze_1);
        // auto res_ti_2 = std::make_shared<opset4::Result>(unsqueeze_2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollTensorIteratorRNNCellSingleIteration) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset4::RNNCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(rnn_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze_1 = std::make_shared<opset4::Squeeze>(X, axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128}, b_val);

        auto rnn_cell_1 = std::make_shared<opset4::RNNCell>(squeeze_1, Y, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<opset4::Unsqueeze>(rnn_cell_1, axis);
        auto res_ti_1 = std::make_shared<opset4::Result>(unsqueeze_1);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollTensorIteratorLSTMCellSingleIterationSingleIteration) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset4::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(lstm_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        // auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y, Z});

        ngraph::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze_1 = std::make_shared<opset4::Squeeze>(X, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell_1 = std::make_shared<opset4::LSTMCell>(squeeze_1, Y, Z, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<opset4::Unsqueeze>(lstm_cell_1, axis);
        auto res_ti_1 = std::make_shared<opset4::Result>(unsqueeze_1);
        // auto res_ti_2 = std::make_shared<opset4::Result>(unsqueeze_2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1}, ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

void collect_legacy_tensor_names(const std::shared_ptr<ov::Model>& model, std::vector<std::string>& holder) {
    for (const auto& op : model->get_ordered_ops()) {
        for (const auto& out : op->outputs()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            auto tensor_name = ov::descriptor::get_ov_tensor_legacy_name(out.get_tensor());
            OPENVINO_SUPPRESS_DEPRECATED_END
            if (!tensor_name.empty() && ov::as_type_ptr<opset8::Result>(op))
                holder.emplace_back(tensor_name);
        }
    }
}

// this test checks that Unroll transformation doesn't insert new tensor names
// (legacy m_tensor_name, not new m_tensor_names) to the graph,
// when TI is not connected to Result operations directly.
// original net:                  Params -> GRUSequence -> Results
// after SeqToTI transformation:  Params -> TI -> Unsqueeze -> Results
// after UnrollTI transformation: Params -> [Unrolled TI] -> Unsqueeze -> Results
// No new tensor names after [UnrolledTI]
TEST(TransformationTests, CheckTensorNamesAfterConvertToTIAndUnrolling) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto X = std::make_shared<opset8::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset8::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto seq_lengths = ngraph::opset8::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384, 16}, w_val);
        auto R = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384, 128}, r_val);
        auto B = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384}, b_val);

        auto rnn_sequence = std::make_shared<opset8::GRUSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset8::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset8::Result>(rnn_sequence->output(1));

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{Y_out, Ho}, ngraph::ParameterVector{X, Y});
    }

    std::vector<std::string> names_before;
    collect_legacy_tensor_names(f, names_before);

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();  // inserts Unsqueeze after TI
    m.register_pass<ov::pass::UnrollTensorIterator>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::vector<std::string> names_after;
    collect_legacy_tensor_names(f, names_after);

    EXPECT_EQ(names_before, names_after);
}

// this test checks that Unroll transformation inserts new tensor names
// (legacy m_tensor_name, not new m_tensor_names) to the graph,
// when TI is connected to Result operations directly.
// original net:                  Params -> TI -> Results
// after UnrollTI transformation: Params -> [Unrolled TI] - tensor names -> Results
TEST(TransformationTests, CheckTensorNamesAfterUnrolling) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset4::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(lstm_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();

        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("TensorIterator");

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1, res_ti_2},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    std::vector<std::string> names_before;
    collect_legacy_tensor_names(f, names_before);

    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::UnrollTensorIterator>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::vector<std::string> names_after;
    collect_legacy_tensor_names(f, names_after);

    EXPECT_NE(names_before, names_after);
    EXPECT_EQ(names_after.size(), 2);
    EXPECT_EQ(names_after[0], "TensorIterator.0");
    EXPECT_EQ(names_after[1], "TensorIterator.1");
}
