// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/fuse_transpose_brgemm.hpp"
#include "common_test_utils/common_utils.hpp"
#include "subgraph_matmul.hpp"
#include "subgraph_lowered.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string FuseTransposeBrgemmTests::getTestCaseName(testing::TestParamInfo<fuseTransposeBrgemmParams> obj) {
    std::vector<PartialShape> input_shapes(2);
    PartialShape master_shape;
    size_t transpose_position;
    std::tie(input_shapes, master_shape, transpose_position) = obj.param;
    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::partialShape2str({input_shapes[0]}) << "_";
    result << "IS[1]=" << CommonTestUtils::partialShape2str({input_shapes[1]}) << "_";
    result << "MS=" << CommonTestUtils::partialShape2str({master_shape}) << "_";
    result << "Pos=" << transpose_position << "_";
    return result.str();
}

void FuseTransposeBrgemmTests::SetUp() {
    LoweringTests::SetUp();
    std::vector<PartialShape> input_shapes(2);
    size_t transpose_position;
    std::tie(input_shapes, master_shape, transpose_position) = this->GetParam();

    snippets_function = std::make_shared<Transpose0213MatMulLoweredFunction>(input_shapes, transpose_position);
}

TEST_P(FuseTransposeBrgemmTests, FuseTransposeMatmul) {
    auto subgraph = getLoweredSubgraph(snippets_function->getOriginal(), master_shape);
    function = subgraph->body_ptr();
    function_ref = snippets_function->getLowered();
}

namespace FuseTransposeBrgemmTestsInstantiation {
using ov::Shape;
std::vector<fuseTransposeBrgemmParams> test_params{
        {{{1, 49, 2, 23}, {2, 2, 23, 39}}, {2, 2, 49, 23}, 0},
        {{{1, 2, 49, 23}, {2, 23, 1, 39}}, {2, 2, 49, 39}, 1},
        {{{1, 2, 49, 23}, {2, 2, 23, 39}}, {2, 2, 49, 39}, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FuseTransposeMatMul, FuseTransposeBrgemmTests,
                         ::testing::ValuesIn(test_params),
                         FuseTransposeBrgemmTests::getTestCaseName);

} // namespace FuseTransposeBrgemmTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov