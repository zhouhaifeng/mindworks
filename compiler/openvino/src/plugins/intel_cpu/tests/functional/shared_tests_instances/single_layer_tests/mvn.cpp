// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mvn.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<bool> emptyAcrossChannels = {{}};
const std::vector<ngraph::AxisSet> emptyReductionAxes = {{}};

const std::vector<std::vector<size_t>> inputShapes = {
    {8},
    {1, 16},
    {3, 19},
    {1, 32, 17},
    {1, 37, 9},
    {1, 16, 5, 8},
    {2, 19, 5, 10},
    {7, 32, 2, 8},
    {5, 8, 3, 5},
    {4, 41, 6, 9},
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
};

const std::vector<bool> acrossChannels = {
    true,
    false
};

const std::vector<bool> normalizeVariance = {
    true,
    false
};

const std::vector<double> epsilon = {
    0.000000001
};

std::vector<InferenceEngine::Precision> dataPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32
};

const auto MvnAcrossChannels = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(dataPrecisions),
    ::testing::ValuesIn(emptyReductionAxes),
    ::testing::ValuesIn(acrossChannels),
    ::testing::ValuesIn(normalizeVariance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto MvnReductionAxes = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 17}, {1, 3, 8, 9}}),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::ValuesIn(std::vector<ngraph::AxisSet>{{1, 2, 3}, {2, 3}}),
    ::testing::ValuesIn(emptyAcrossChannels),
    ::testing::ValuesIn(normalizeVariance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN_AcrossChannels, Mvn1LayerTest, MvnAcrossChannels, Mvn1LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN_ReductionAxes, Mvn1LayerTest, MvnReductionAxes, Mvn1LayerTest::getTestCaseName);


std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::I64
};

const std::vector<std::string> epsMode = {
    "inside_sqrt",
    "outside_sqrt"
};

const std::vector<float> epsilonF = {
    0.0001f
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_5D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 7, 8}, {1, 3, 8, 9, 49}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 3, 4}, {2, 3, 4}, {-3, -2, -1}, {-1, -4, -2, -3}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_4D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 17}, {1, 3, 8, 9}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_3D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 32, 17}, {1, 37, 9}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2}, {2}, {-1}, {-1, -2}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_2D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 5}, {2, 55}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_1D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3}, {9}, {55}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_3D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 32, 17}, {1, 37, 9}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2}, {0}, {1}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_4D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 5, 8}, {2, 19, 5, 10}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2, 3}, {0, 1, 2}, {0, 3}, {0}, {1}, {2}, {3}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_10D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3, 5, 4, 2, 6, 5, 3, 2, 1}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 5, 8, 9}, {0, 1, 2, 3}, {0, 1, 2}, {0, 3}, {0}, {3}, {5}, {9}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);
