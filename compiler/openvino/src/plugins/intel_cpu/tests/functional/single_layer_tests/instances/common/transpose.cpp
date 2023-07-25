// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/transpose.hpp"
#include "shared_test_classes/single_layer/transpose.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Transpose {
std::map<std::string, std::string> additional_config;

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {}, {}, {}};
const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {}, {}, {}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        Precision::I8,
        Precision::FP32
};

const std::vector<std::vector<size_t>> inputOrderPerChannels4D = {
        std::vector<size_t>{0, 1, 2, 3},
        std::vector<size_t>{0, 2, 1, 3},
        std::vector<size_t>{1, 0, 2, 3},
        std::vector<size_t>{},
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nchw,
};

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC16_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC16()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC32()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::ValuesIn(CPUParams4D)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_Transpose, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D()),
                                 ::testing::ValuesIn(inputOrder4D()),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC16_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC16()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_staticShapes4DC32_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4DC32()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(cpuParams_nhwc)),
                         TransposeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamicShapes4D_PermutePerChannels, TransposeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShapes4D()),
                                 ::testing::ValuesIn(inputOrderPerChannels4D),
                                 ::testing::ValuesIn(netPrecisionsPerChannels()),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(additional_config),
                                 ::testing::Values(CPUSpecificParams{})),
                         TransposeLayerCPUTest::getTestCaseName);

} // namespace Transpose
} // namespace CPULayerTestsDefinitions