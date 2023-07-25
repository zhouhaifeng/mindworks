// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unary_ops.hpp"

using Types = ::testing::Types<UnaryOperatorType<ov::op::v4::SoftPlus, ngraph::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_atrribute, UnaryOperatorVisitor, Types, UnaryOperatorTypeName);
