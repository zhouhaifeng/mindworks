// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "binary_ops.hpp"
#include "ngraph/opsets/opset1.hpp"

using Type = ::testing::Types<BinaryOperatorType<ov::op::v1::Add, ngraph::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
