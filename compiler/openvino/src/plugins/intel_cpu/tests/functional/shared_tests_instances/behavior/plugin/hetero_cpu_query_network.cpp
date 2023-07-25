// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/hetero_query_network.hpp"

using namespace HeteroTests;

namespace HeteroTests {

TEST_P(HeteroQueryNetworkTest, HeteroSinglePlugin) {
    std::string deviceName = GetParam();
    RunTest(deviceName);
}

INSTANTIATE_TEST_CASE_P(
        HeteroCpu,
        HeteroQueryNetworkTest,
        ::testing::Values(
                std::string("HETERO:CPU")));

} // namespace HeteroTests
