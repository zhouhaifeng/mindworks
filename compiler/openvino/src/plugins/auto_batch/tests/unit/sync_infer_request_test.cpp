// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"
#include "transformations/utils/utils.hpp"
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

using AutoBatchRequestTestParams = std::tuple<uint32_t,              // batch_size
                                              ov::element::Type_t>;  // data type

class AutoBatchRequestTest : public ::testing::TestWithParam<AutoBatchRequestTestParams> {
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

    std::shared_ptr<NiceMock<MockISyncInferRequest>> m_sync_infer_request_with_batch;

    std::shared_ptr<NiceMock<MockIAsyncInferRequest>> m_async_infer_request_with_batch;

    std::shared_ptr<ov::threading::ImmediateExecutor> m_executor;

    std::shared_ptr<CompiledModel::WorkerInferRequest> workerRequestPtr;

    uint32_t m_batch_size;
    ov::element::Type_t m_element_type;

    std::vector<std::shared_ptr<SyncInferRequest>> m_auto_batch_infer_requests;

    std::vector<ov::ProfilingInfo> m_profiling_info;

    static std::string getTestCaseName(testing::TestParamInfo<AutoBatchRequestTestParams> obj) {
        uint32_t batch_size;
        ov::element::Type_t element_type;
        std::tie(batch_size, element_type) = obj.param;

        std::string res;
        res = "batch_size_" + std::to_string(batch_size);
        res += "_element_type_" + std::to_string(static_cast<int>(element_type));
        return res;
    }

    void TearDown() override {
        m_profiling_info.clear();
        m_auto_batch_infer_requests.clear();
        m_auto_batch_plugin.reset();
        m_model.reset();
        m_core.reset();
        m_i_compile_model_without_batch.reset();
        m_compile_model_without_batch = {};
        m_i_compile_model_with_batch.reset();
        m_compile_model_with_batch = {};
        m_auto_batch_compile_model.reset();
        m_sync_infer_request_with_batch.reset();
        m_async_infer_request_with_batch.reset();
        m_executor.reset();
        clear_worker();
        workerRequestPtr.reset();
    }

    void SetUp() override {
        std::tie(m_batch_size, m_element_type) = this->GetParam();
        std::vector<size_t> inputShape = {1, 3, 24, 24};
        m_model = ngraph::builder::subgraph::makeMultiSingleConv(inputShape, m_element_type);
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

        m_i_compile_model_with_batch = std::make_shared<NiceMock<MockICompiledModel>>(m_model, m_auto_batch_plugin);
        m_compile_model_with_batch = {m_i_compile_model_with_batch, {}};

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

        m_sync_infer_request_with_batch =
            std::make_shared<NiceMock<MockISyncInferRequest>>(m_i_compile_model_with_batch);

        m_executor = std::make_shared<ov::threading::ImmediateExecutor>();

        m_async_infer_request_with_batch =
            std::make_shared<NiceMock<MockIAsyncInferRequest>>(m_sync_infer_request_with_batch, m_executor, nullptr);

        m_profiling_info = {};
    }

    void create_worker(int batch_size) {
        workerRequestPtr = std::make_shared<CompiledModel::WorkerInferRequest>();

        workerRequestPtr->_infer_request_batched = {m_async_infer_request_with_batch, {}};
        workerRequestPtr->_batch_size = batch_size;
        workerRequestPtr->_completion_tasks.resize(workerRequestPtr->_batch_size);
        workerRequestPtr->_infer_request_batched->set_callback([this](std::exception_ptr exceptionPtr) mutable {
            if (exceptionPtr)
                workerRequestPtr->_exception_ptr = exceptionPtr;
        });
        workerRequestPtr->_thread = std::thread([] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        return;
    }

    void clear_worker() {
        workerRequestPtr->_infer_request_batched = {};
        workerRequestPtr->_completion_tasks.clear();
        workerRequestPtr->_thread.join();
    }

    void prepare_input(std::shared_ptr<ov::Model>& model, int batch_size) {
        const auto& params = model->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            m_batched_inputs.insert(ov::op::util::get_ie_output_name(params[i]->output(0)));
        }
        const auto& results = model->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            const auto& output = results[i];
            const auto& node = output->input_value(0);
            m_batched_outputs.insert(
                ov::op::util::get_ie_output_name(ov::Output<const ov::Node>(node.get_node(), node.get_index())));
        }
    }
};

TEST_P(AutoBatchRequestTest, AutoBatchRequestCreateTestCase) {
    prepare_input(m_model, m_batch_size);
    create_worker(m_batch_size);

    for (uint32_t batch_id = 0; batch_id < m_batch_size; batch_id++) {
        auto req = std::make_shared<SyncInferRequest>(m_auto_batch_compile_model,
                                                      workerRequestPtr,
                                                      batch_id,
                                                      m_batch_size,
                                                      m_batched_inputs,
                                                      m_batched_outputs);
        EXPECT_NE(req, nullptr);
        m_auto_batch_infer_requests.emplace_back(req);
    }
}

TEST_P(AutoBatchRequestTest, AutoBatchRequestCopyInputTensorTestCase) {
    prepare_input(m_model, m_batch_size);
    create_worker(m_batch_size);

    auto req = std::make_shared<SyncInferRequest>(m_auto_batch_compile_model,
                                                  workerRequestPtr,
                                                  0,
                                                  m_batch_size,
                                                  m_batched_inputs,
                                                  m_batched_outputs);
    EXPECT_NE(req, nullptr);
    m_auto_batch_infer_requests.emplace_back(req);

    EXPECT_NO_THROW(req->copy_inputs_if_needed());
}

TEST_P(AutoBatchRequestTest, AutoBatchRequestCopyOutputTensorTestCase) {
    prepare_input(m_model, m_batch_size);
    create_worker(m_batch_size);

    auto req = std::make_shared<SyncInferRequest>(m_auto_batch_compile_model,
                                                  workerRequestPtr,
                                                  0,
                                                  m_batch_size,
                                                  m_batched_inputs,
                                                  m_batched_outputs);
    EXPECT_NE(req, nullptr);
    m_auto_batch_infer_requests.emplace_back(req);

    EXPECT_NO_THROW(req->copy_outputs_if_needed());
}

TEST_P(AutoBatchRequestTest, AutoBatchRequestGetProfilingInfoTestCase) {
    prepare_input(m_model, m_batch_size);
    create_worker(m_batch_size);

    auto req = std::make_shared<SyncInferRequest>(m_auto_batch_compile_model,
                                                  workerRequestPtr,
                                                  0,
                                                  m_batch_size,
                                                  m_batched_inputs,
                                                  m_batched_outputs);
    EXPECT_NE(req, nullptr);

    ON_CALL(*m_sync_infer_request_with_batch, get_profiling_info()).WillByDefault(Return(m_profiling_info));

    EXPECT_NO_THROW(req->get_profiling_info());
}

std::vector<ov::element::Type_t> element_type{ov::element::Type_t::f16,
                                              ov::element::Type_t::f32,
                                              ov::element::Type_t::f64,
                                              ov::element::Type_t::i8,
                                              ov::element::Type_t::i16,
                                              ov::element::Type_t::i32,
                                              ov::element::Type_t::i64,
                                              ov::element::Type_t::u8,
                                              ov::element::Type_t::u16,
                                              ov::element::Type_t::u32,
                                              ov::element::Type_t::u64};
const std::vector<uint32_t> batch_size{1, 8, 16, 32, 64, 128};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         AutoBatchRequestTest,
                         ::testing::Combine(::testing::ValuesIn(batch_size), ::testing::ValuesIn(element_type)),
                         AutoBatchRequestTest::getTestCaseName);