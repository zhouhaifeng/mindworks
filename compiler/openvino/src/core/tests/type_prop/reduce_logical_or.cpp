// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

using Type = ::testing::Types<op::v1::ReduceLogicalOr>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_logical_or, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_logical_or_et, ReduceLogicalTest, Type);
