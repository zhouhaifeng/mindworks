// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unary_ops.hpp"

using Type = ::testing::Types<ngraph::op::Floor>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_floor, UnaryOperator, Type);
