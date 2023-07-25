// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"
#include "sync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {
class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& request,
                      const ov::SoPtr<ov::IAsyncInferRequest>& request_without_batch,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    void infer_thread_unsafe() override;

    virtual ~AsyncInferRequest();

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    std::shared_ptr<ov::autobatch_plugin::SyncInferRequest> m_sync_request;

    ov::SoPtr<ov::IAsyncInferRequest> m_request_without_batch;
};
}  // namespace autobatch_plugin
}  // namespace ov