// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include <functional>
#include <memory>
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace cldnn {

class ICompilationContext {
public:
    using Task = std::function<void()>;
    virtual void push_task(kernel_impl_params key, Task&& task) = 0;
    virtual void remove_keys(std::vector<kernel_impl_params>&& keys) = 0;
    virtual ~ICompilationContext() = default;
    virtual bool is_stopped() = 0;
    virtual void cancel() = 0;

    static std::unique_ptr<ICompilationContext> create(ov::threading::IStreamsExecutor::Config task_executor_config);
};

}  // namespace cldnn
