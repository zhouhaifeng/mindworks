// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        ov::Shape,                   // Input 0 Shape
        ov::Shape,                   // Input 1 Shape
        ov::Shape,                   // Input 2 Shape
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> ThreeInputsEltwiseParams;

typedef std::tuple<
    InputShape,                   // Input 0 Shape
    InputShape,                   // Input 1 Shape
    InputShape,                   // Input 2 Shape
    size_t,                      // Expected num nodes
    size_t,                      // Expected num subgraphs
    std::string                  // Target Device
    > ThreeInputsEltwiseDynamicParams;

class ThreeInputsEltwise : public testing::WithParamInterface<ov::test::snippets::ThreeInputsEltwiseParams>,
                   virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::ThreeInputsEltwiseParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov
