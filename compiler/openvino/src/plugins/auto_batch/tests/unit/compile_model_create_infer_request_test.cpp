// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;

using namespace ov::mock_autobatch_plugin;

using CreateInferRequestTestParams = std::tuple<int,   // batch_size
                                                int>;  // inferReq number

class CompileModelCreateInferRequestTest : public ::testing::TestWithParam<CreateInferRequestTestParams> {
public:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<NiceMock<MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_auto_batch_plugin;

    std::shared_ptr<NiceMock<MockICompiledModel>> m_i_compile_model_without_batch;
    ov::SoPtr<ov::ICompiledModel> m_compile_model_without_batch;

    std::shared_ptr<NiceMock<MockICompiledModel>> m_i_compile_model_with_batch;
    ov::SoPtr<ov::ICompiledModel> m_compile_model_with_batch;

    ov::AnyMap m_config;
    DeviceInformation m_device_info;
    std::set<std::string> m_batched_inputs;
    std::set<std::string> m_batched_outputs;
    ov::SoPtr<ov::IRemoteContext> m_remote_context;

    std::shared_ptr<MockAutoBatchCompileModel> m_auto_batch_compile_model;

    std::shared_ptr<NiceMock<MockISyncInferRequest>> m_sync_infer_request;

    std::shared_ptr<ov::threading::ImmediateExecutor> m_executor;

    uint32_t m_batch_size;
    int m_infer_request_num;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CreateInferRequestTestParams> obj) {
        int batch_size;
        int infer_num;
        std::tie(batch_size, infer_num) = obj.param;

        std::string res;
        res = "batch_size_" + std::to_string(batch_size);
        res += "_infer_num_" + std::to_string(infer_num);
        return res;
    }

    void TearDown() override {
        m_auto_batch_plugin.reset();
        m_model.reset();
        m_core.reset();
        m_i_compile_model_without_batch.reset();
        m_compile_model_without_batch = {};
        m_i_compile_model_with_batch.reset();
        m_compile_model_with_batch = {};
        m_auto_batch_compile_model.reset();
        m_sync_infer_request.reset();
        m_executor.reset();
    }

    void SetUp() override {
        std::tie(m_batch_size, m_infer_request_num) = this->GetParam();
        m_model = ngraph::builder::subgraph::makeMultiSingleConv();
        m_core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());

        m_auto_batch_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());

        m_auto_batch_plugin->set_core(m_core);
        m_i_compile_model_without_batch = std::make_shared<NiceMock<MockICompiledModel>>(m_model, m_auto_batch_plugin);
        m_compile_model_without_batch = {m_i_compile_model_without_batch, {}};

        m_config = {{"AUTO_BATCH_TIMEOUT", "200"}};

        m_device_info = {"CPU", {}, m_batch_size};
        m_batched_inputs = {"Parameter_0"};
        m_batched_outputs = {"Convolution_20"};

        if (m_batch_size > 1) {
            m_i_compile_model_with_batch = std::make_shared<NiceMock<MockICompiledModel>>(m_model, m_auto_batch_plugin);
            m_compile_model_with_batch = {m_i_compile_model_with_batch, {}};
        }

        ASSERT_NO_THROW(m_auto_batch_compile_model =
                            std::make_shared<MockAutoBatchCompileModel>(m_model->clone(),
                                                                        m_auto_batch_plugin,
                                                                        m_config,
                                                                        m_device_info,
                                                                        m_batched_inputs,
                                                                        m_batched_outputs,
                                                                        m_compile_model_with_batch,
                                                                        m_compile_model_without_batch,
                                                                        m_remote_context));

        m_sync_infer_request = std::make_shared<NiceMock<MockISyncInferRequest>>(m_i_compile_model_without_batch);

        m_executor = std::make_shared<ov::threading::ImmediateExecutor>();

        ON_CALL(*m_i_compile_model_without_batch, create_infer_request()).WillByDefault([this]() {
            return std::make_shared<NiceMock<MockIAsyncInferRequest>>(m_sync_infer_request, m_executor, nullptr);
        });

        EXPECT_CALL(*m_auto_batch_compile_model, create_sync_infer_request())
            .WillRepeatedly(Return(m_sync_infer_request));
    }
};

TEST_P(CompileModelCreateInferRequestTest, CreateInferRequestTestCases) {
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> inferReqs;
    std::shared_ptr<ov::IAsyncInferRequest> inferReq;
    for (int i = 0; i < m_infer_request_num; i++) {
        EXPECT_NO_THROW(inferReq = m_auto_batch_compile_model->create_infer_request());
        EXPECT_NE(inferReq, nullptr);
        inferReqs.push_back(inferReq);
    }
    inferReqs.clear();
}

const std::vector<int> requests_num{1, 8, 16, 64};
const std::vector<int> batch_size{1, 8, 16, 32, 128, 256};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         CompileModelCreateInferRequestTest,
                         ::testing::Combine(::testing::ValuesIn(batch_size), ::testing::ValuesIn(requests_num)),
                         CompileModelCreateInferRequestTest::getTestCaseName);