// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/function.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph/function.hpp>
#include <common_test_utils/test_constants.hpp>
#include <cpp/ie_cnn_network.h>
#include "gtest/gtest.h"
#include "functional_test_utils/crash_handler.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "base/behavior_test_utils.hpp"
#include <ie_core.hpp>

namespace BehaviorTestsDefinitions {
typedef std::tuple<
        std::string,         // Target device name
        std::vector<int>>    // Order
HoldersParams;

class HoldersTest : public BehaviorTestsUtils::IEPluginTestBase,
                    public ::testing::WithParamInterface<HoldersParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<HoldersParams> obj);
    void SetUp() override;

protected:
    std::vector<int> order;
    std::shared_ptr<ngraph::Function> function;
};

using HoldersTestImportNetwork = HoldersTest;

class HoldersTestOnImportedNetwork : public BehaviorTestsUtils::IEPluginTestBase,
                                     public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;

protected:
    std::shared_ptr<ngraph::Function> function;
};

}  // namespace BehaviorTestsDefinitions
