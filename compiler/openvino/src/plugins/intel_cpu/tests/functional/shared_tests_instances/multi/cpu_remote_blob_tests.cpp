// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "multi/multi_remote_blob_tests.hpp"
#include "common_test_utils/test_constants.hpp"

const std::vector<DevicesNamesAndSupportTuple> device_names_and_support_for_remote_blobs {
        {{CPU}, false, {}}, // CPU via MULTI
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_RemoteBlobCPU, MultiDevice_SupportTest,
        ::testing::ValuesIn(device_names_and_support_for_remote_blobs), MultiDevice_SupportTest::getTestCaseName);
