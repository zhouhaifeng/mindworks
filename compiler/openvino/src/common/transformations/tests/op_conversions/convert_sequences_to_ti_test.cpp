// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertLSTMSequenceToTensorIterator) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 512}, b_val);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X,
                                                                   Y,
                                                                   Z,
                                                                   seq_lengths,
                                                                   W,
                                                                   R,
                                                                   B,
                                                                   128,
                                                                   op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{Y_out, Ho, Co}, ngraph::ParameterVector{X, Y, Z});

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));

        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze_y);

        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));

        auto body =
            std::make_shared<Function>(OutputVector{Y_out, Ho, Co}, ParameterVector{Xi, Yi, Zi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_Y, res_ti_H, res_ti_C},
                                                   ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertLSTMSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 512}, b_val);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X,
                                                                   Y,
                                                                   Z,
                                                                   seq_lengths,
                                                                   W,
                                                                   R,
                                                                   B,
                                                                   128,
                                                                   op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{Y_out, Ho, Co}, ngraph::ParameterVector{X, Y, Z});

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));

        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze_y);

        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));

        auto body =
            std::make_shared<Function>(OutputVector{Y_out, Ho, Co}, ParameterVector{Xi, Yi, Zi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_Y, res_ti_H, res_ti_C},
                                                   ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertRNNSequenceToTensorIterator) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128}, b_val);

        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{Y_out, Ho}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertRNNSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{128, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{128, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze_x, Yi, W, R, B, 128);
        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_Y, res_ti_H}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertRNNSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128}, b_val);

        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{Y_out, Ho}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertRNNSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, axis_1);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, axis_1);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{128, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{128, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, axis_1);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y =
            std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), axis_1));
        auto res_ti_H =
            std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), axis_1));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_Y, res_ti_H}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGRUSequenceToTensorIterator) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384}, b_val);

        auto rnn_sequence = std::make_shared<opset5::GRUSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{Y_out, Ho}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{384}, b_val);

        auto rnn_cell = std::make_shared<opset5::GRUCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_Y, res_ti_H}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGRUSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 384}, b_val);

        auto rnn_sequence = std::make_shared<opset5::GRUSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{Y_out, Ho}, ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{384}, b_val);

        auto rnn_cell = std::make_shared<opset5::GRUCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_Y, res_ti_H}, ngraph::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
