// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        ov::Shape,                       // Input 0 Shape
        int,                             // Axis
        size_t,                          // Expected num nodes
        size_t,                          // Expected num subgraphs
        std::string                      // Target Device
> SoftmaxParams;

typedef std::tuple<
        std::pair<ov::Shape, ov::Shape>,  // Input Shapes
        int,                              // Axis
        size_t,                           // Expected num nodes
        size_t,                           // Expected num subgraphs
        std::string                       // Target Device
> AddSoftmaxParams;

class Softmax : public testing::WithParamInterface<ov::test::snippets::SoftmaxParams>,
                virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::SoftmaxParams> obj);

protected:
    void SetUp() override;
};

class AddSoftmax : public testing::WithParamInterface<ov::test::snippets::AddSoftmaxParams>,
                   virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddSoftmaxParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov