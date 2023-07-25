// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

enum ShapeMode {
    DYNAMIC,
    STATIC,
    BOTH
};

extern ShapeMode shapeMode;

using ReadIRParams = std::tuple<
        std::pair<std::string, std::string>, // pair<ir_path, cache_path>
        std::string,                         // Target Device
        ov::AnyMap>;                         // Plugin Config

class ReadIRTest : public testing::WithParamInterface<ReadIRParams>,
                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj);
    void query_model() override;
    void import_export();
    std::vector<ov::Tensor> calculate_refs() override;

protected:
    void SetUp() override;

private:
    std::string path_to_model, path_to_cache;
    std::vector<std::pair<std::string, size_t>> ocurance_in_models;
};
} // namespace subgraph
} // namespace test
} // namespace ov
