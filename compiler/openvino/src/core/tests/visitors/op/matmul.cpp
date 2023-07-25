// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, matmul_op) {
    NodeBuilder::get_ops().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{0, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 0});

    bool transpose_a = true;
    bool transpose_b = true;

    auto matmul = make_shared<opset1::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op2) {
    NodeBuilder::get_ops().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{10, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    bool transpose_a = false;
    bool transpose_b = false;

    auto matmul = make_shared<opset1::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op3) {
    NodeBuilder::get_ops().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 10});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    bool transpose_a = true;
    bool transpose_b = false;

    auto matmul = make_shared<opset1::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op4) {
    NodeBuilder::get_ops().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{3, 2, 2, 1});

    auto matmul = make_shared<opset1::MatMul>(A, B);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op5) {
    NodeBuilder::get_ops().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 10});

    auto matmul = make_shared<opset1::MatMul>(A, B);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, matmul_op6) {
    NodeBuilder::get_ops().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2048});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2048, 1000});

    auto matmul = make_shared<opset1::MatMul>(A, B);
    NodeBuilder builder(matmul, {A, B});
    auto g_matmul = ov::as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}
