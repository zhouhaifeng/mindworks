// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/auto_unit_test.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"
#include "openvino/runtime/auto/properties.hpp"

using namespace ov::mock_auto_plugin;

using ConfigParams = std::tuple<std::vector<std::tuple<std::string, bool>>, int, bool, bool, bool, bool>;

class AutoRuntimeFallback : public tests::AutoTest,
                            public ::testing::TestWithParam<ConfigParams> {
public:
    ov::SoPtr<ov::MockCompiledModel>  mockExeNetworkGPU_1;
    ov::SoPtr<ov::MockCompiledModel>  mockExeNetworkVPU;

    std::shared_ptr<NiceMock<ov::MockSyncInferRequest>>     inferReqInternalGPU_1;
    std::shared_ptr<NiceMock<ov::MockSyncInferRequest>>     inferReqInternalVPU;

    std::shared_ptr<NiceMock<ov::MockCompiledModel>>     mockIExeNetGPU_1;
    std::shared_ptr<NiceMock<ov::MockCompiledModel>>     mockIExeNetVPU;

    std::shared_ptr<ov::MockAsyncInferRequest>     mockInferrequest;
    std::shared_ptr<ov::MockAsyncInferRequest>     mockInferrequestGPU_0;
    std::shared_ptr<ov::MockAsyncInferRequest>     mockInferrequestGPU_1;
    std::shared_ptr<ov::MockAsyncInferRequest>     mockInferrequestVPU;

    std::shared_ptr<ov::threading::ImmediateExecutor>     mockExecutor;
    std::shared_ptr<ov::threading::ImmediateExecutor>     mockExecutorGPU_0;
    std::shared_ptr<ov::threading::ImmediateExecutor>     mockExecutorGPU_1;
    std::shared_ptr<ov::threading::ImmediateExecutor>     mockExecutorVPU;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::vector<std::tuple<std::string, bool>> targetDevices;
        int loadNetworkNum;
        bool enableRumtimeFallback;
        bool expectThrow;
        bool loadNetworkFail;
        bool generateWorkersFail;
        std::tie(targetDevices, loadNetworkNum, enableRumtimeFallback, expectThrow, loadNetworkFail, generateWorkersFail) = obj.param;
        std::ostringstream result;
        result << "auto_runtime_fallback_";
        for (auto deviceInfo : targetDevices) {
            std::string deviceName;
            bool ifThrow;
            std::tie(deviceName, ifThrow) = deviceInfo;
            result << deviceName << "_";
            if (ifThrow)
                result << "true_";
            else
                result << "false_";
        }
        if (enableRumtimeFallback)
            result << "enableRuntimeFallback";
        else
            result << "disableRuntimeFallback";
        if (loadNetworkFail)
            result << "loadNetworkFail";
        if (generateWorkersFail)
            result << "generateWorkersFail";
        return result.str();
    }

    void TearDown() override {
        mockExeNetworkGPU_1 = {};
        inferReqInternalGPU_1.reset();
        inferReqInternalVPU.reset();
        mockIExeNetGPU_1.reset();
        mockIExeNetVPU.reset();
        mockIExeNetGPU_1.reset();
        mockIExeNetVPU.reset();
        mockExecutor.reset();
        mockExecutorGPU_0.reset();
        mockExecutorGPU_1.reset();
        mockExecutorVPU.reset();
    }

    void SetUp() override {
        // prepare extra mockExeNetwork
        mockIExeNetGPU_1 = std::make_shared<NiceMock<ov::MockCompiledModel>>(model, plugin);
        mockExeNetworkGPU_1 = {mockIExeNetGPU_1, {}};

        mockIExeNetVPU = std::make_shared<NiceMock<ov::MockCompiledModel>>(model, plugin);
        mockExeNetworkVPU = {mockIExeNetVPU, {}};

        // prepare mockicore and cnnNetwork for loading
        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                    ::testing::Matcher<const std::string&>(StrEq("GPU.0")), _)).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkActual; }));
        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                    ::testing::Matcher<const std::string&>(StrEq("GPU.1")), _)).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkGPU_1; }));
        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_KEEMBAY)), _)).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkVPU; }));

        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                    (_))).WillByDefault(Return(mockExeNetwork));

        mockExecutor = std::make_shared<ov::threading::ImmediateExecutor>();

        mockExecutorGPU_0 = std::make_shared<ov::threading::ImmediateExecutor>();

        inferReqInternalGPU_1 = std::make_shared<NiceMock<ov::MockSyncInferRequest>>(mockIExeNetGPU_1);
        mockExecutorGPU_1 = std::make_shared<ov::threading::ImmediateExecutor>();
        ON_CALL(*mockIExeNetGPU_1, get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
           .WillByDefault(Return(optimalNum));

        inferReqInternalVPU = std::make_shared<NiceMock<ov::MockSyncInferRequest>>(mockIExeNetVPU);
        mockExecutorVPU = std::make_shared<ov::threading::ImmediateExecutor>();
        ON_CALL(*mockIExeNetVPU, get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
           .WillByDefault(Return(optimalNum));
    }
};

using AutoCTPUTRuntimeFallback = AutoRuntimeFallback;

TEST_P(AutoRuntimeFallback, releaseResource) {
    std::string targetDev;
    std::vector<std::tuple<std::string, bool>> targetDevices;
    int loadNetworkNum;
    bool enableRumtimeFallback;
    bool expectThrow;
    bool loadNetworkFail;
    bool generateWorkersFail;
    std::tie(targetDevices, loadNetworkNum, enableRumtimeFallback, expectThrow, loadNetworkFail, generateWorkersFail) = this->GetParam();
    if (loadNetworkFail) {
        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
            ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
            _)).WillByDefault(Throw(ov::Exception{"compile model error"}));
    }
    for (auto& deviceInfo : targetDevices) {
        std::string deviceName;
        bool ifThrow;
        std::tie(deviceName, ifThrow) = deviceInfo;
        targetDev += deviceName;
        targetDev += ((deviceInfo == targetDevices.back()) ? "" : ",");
        if (deviceName == "CPU") {
            mockInferrequest = std::make_shared<ov::MockAsyncInferRequest>(
                inferReqInternal, mockExecutor, nullptr, ifThrow);
            ON_CALL(*mockIExeNet.get(), create_infer_request()).WillByDefault(Return(mockInferrequest));
        } else if (deviceName == "GPU.0") {
            mockInferrequestGPU_0 = std::make_shared<ov::MockAsyncInferRequest>(
                inferReqInternalActual, mockExecutorGPU_0, nullptr, ifThrow);
            ON_CALL(*mockIExeNetActual.get(), create_infer_request()).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(0));
                        return mockInferrequestGPU_0; }));
        } else if (deviceName == "GPU.1") {
            if (generateWorkersFail) {
                mockInferrequestGPU_1 = std::make_shared<ov::MockAsyncInferRequest>(
                    inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), create_infer_request()).WillByDefault(Throw(ov::Exception{"error"}));
            } else {
                mockInferrequestGPU_1 = std::make_shared<ov::MockAsyncInferRequest>(
                    inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), create_infer_request()).WillByDefault(InvokeWithoutArgs([this]() {
                            std::this_thread::sleep_for(std::chrono::milliseconds(0));
                            return mockInferrequestGPU_1; }));
            }
        } else if (deviceName == "VPU") {
            mockInferrequestVPU = std::make_shared<ov::MockAsyncInferRequest>(
                inferReqInternalVPU, mockExecutorVPU, nullptr, ifThrow);
            ON_CALL(*mockIExeNetVPU.get(), create_infer_request()).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(0));
                        return mockInferrequestVPU; }));
        } else {
            return;
        }
    }
    plugin->set_device_name("AUTO");
    config.insert(ov::device::priorities(targetDev));
    if (!enableRumtimeFallback) {
        config.insert(ov::intel_auto::enable_runtime_fallback(false));
    }

    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                            ::testing::Matcher<const std::string&>(_),
                            ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(loadNetworkNum);

    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;

    ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
    ASSERT_NO_THROW(infer_request = exeNetwork->create_infer_request());
    if (expectThrow) {
        EXPECT_THROW(infer_request->infer(), ov::Exception);
    } else {
        ASSERT_NO_THROW(infer_request->infer());
    }
}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}}, 2, true, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", true}}, 1, true, false, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}}, 1, true, false, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"CPU", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", false}}, 2, true, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"CPU", true}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", true}}, 2, true, true, false, false},
    // 3 devices
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"VPU", false}}, 1, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPU", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPU", false}}, 3, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPU", true}}, 3, true, true, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", false}}, 2, true, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", true}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", true}}, 3, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", true}}, 3, true, true, false, false},
    // disable RumtimeFallback
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", true}}, 1, false, false, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}}, 1, false, false, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"CPU", false}}, 2, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", false}}, 2, false, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"CPU", true}}, 2, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", true}}, 2, false, true, false, false},
    // 3 devices
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"VPU", false}}, 1, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPU", false}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPU", false}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPU", true}}, 1, false, true, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", false}}, 2, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", false}}, 2, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", false}}, 2, false, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", true}}, 2, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", true}}, 2, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", true}}, 2, false, true, false, false},
    // loadFail and CreateInferRequestFail
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPU", false}}, 3, true, false, true, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPU", false}}, 3, true, false, false, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoRuntimeFallback, AutoRuntimeFallback,
                ::testing::ValuesIn(testConfigs),
           AutoRuntimeFallback::getTestCaseName);

TEST_P(AutoCTPUTRuntimeFallback, ctputDeviceInferFailTest) {
    std::string targetDev;
    std::vector<std::tuple<std::string, bool>> targetDevices; //std::tuple<deviceName, will infer throw exception>
    int loadNetworkNum;
    bool enableRumtimeFallback;
    bool expectThrow;
    bool loadNetworkFail;
    bool generateWorkersFail;
    std::tie(targetDevices, loadNetworkNum, enableRumtimeFallback, expectThrow, loadNetworkFail, generateWorkersFail) = this->GetParam();
    if (loadNetworkFail) {
        ON_CALL(*core, compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
            ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
            _)).WillByDefault(Throw(ov::Exception{"compile model error"}));
    }
    for (auto& deviceInfo : targetDevices) {
        std::string deviceName;
        bool ifThrow;
        std::tie(deviceName, ifThrow) = deviceInfo;
        targetDev += deviceName;
        targetDev += ((deviceInfo == targetDevices.back()) ? "" : ",");
        if (deviceName == "CPU") {
            mockInferrequest = std::make_shared<ov::MockAsyncInferRequest>(
                inferReqInternal, mockExecutor, nullptr, ifThrow);
            ON_CALL(*mockIExeNet.get(), create_infer_request()).WillByDefault(Return(mockInferrequest));
        } else if (deviceName == "GPU.0") {
            mockInferrequestGPU_0 = std::make_shared<ov::MockAsyncInferRequest>(
                inferReqInternalActual, mockExecutorGPU_0, nullptr, ifThrow);
            ON_CALL(*mockIExeNetActual.get(), create_infer_request()).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(0));
                        return mockInferrequestGPU_0; }));
        } else if (deviceName == "GPU.1") {
            if (generateWorkersFail) {
                mockInferrequestGPU_1 = std::make_shared<ov::MockAsyncInferRequest>(
                    inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), create_infer_request()).WillByDefault(Throw(ov::Exception{"error"}));
            } else {
                mockInferrequestGPU_1 = std::make_shared<ov::MockAsyncInferRequest>(
                    inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), create_infer_request()).WillByDefault(InvokeWithoutArgs([this]() {
                            std::this_thread::sleep_for(std::chrono::milliseconds(0));
                            return mockInferrequestGPU_1; }));
            }
        }
    }
    plugin->set_device_name("AUTO");
    config.insert(ov::device::priorities(targetDev));
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    if (!enableRumtimeFallback) {
        config.insert(ov::intel_auto::enable_runtime_fallback(false));
    }

    EXPECT_CALL(*core,
                compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                            ::testing::Matcher<const std::string&>(_),
                            ::testing::Matcher<const ov::AnyMap&>(_)))
        .Times(loadNetworkNum);

    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;

    ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));
    ASSERT_NO_THROW(infer_request = exeNetwork->create_infer_request());
    if (expectThrow) {
        EXPECT_THROW(infer_request->infer(), ov::Exception);
    } else {
        ASSERT_NO_THROW(infer_request->infer());
    }
}

// ConfigParams: targetDevices(deviceName, will infer throw exception), loadNetworkNum, enableRumtimeFallback,
// expectThrow, loadNetworkFail, generateWorkersFail
const std::vector<ConfigParams> testCtputConfigs = {
    ConfigParams{{{"CPU", false}, {"GPU.0", true}, {"GPU.1", true}}, 3, true, false, false, false},
    ConfigParams{{{"CPU", true}, {"GPU.0", false}, {"GPU.1", true}}, 3, true, false, false, false},
    ConfigParams{{{"CPU", true}, {"GPU.0", true}, {"GPU.1", true}}, 3, true, true, false, false},
    // disable RumtimeFallback
    ConfigParams{{{"CPU", false}, {"GPU.0", false}, {"GPU.1", false}}, 3, false, false, false, false},
    ConfigParams{{{"CPU", true}, {"GPU.0", false}, {"GPU.1", false}}, 3, false, true, false, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoCTPUTRuntimeFallback,
                         AutoCTPUTRuntimeFallback,
                         ::testing::ValuesIn(testCtputConfigs),
                         AutoCTPUTRuntimeFallback::getTestCaseName);