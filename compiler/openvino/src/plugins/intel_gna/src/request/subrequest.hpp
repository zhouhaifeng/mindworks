// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>

#include "request_status.hpp"

namespace ov {
namespace intel_gna {
namespace request {

/**
 * @interface Interface representing subrequest of worker.
 */
class Subrequest {
public:
    /**
     * @brief Callback invoked by enqueue operation.
     * @return request id
     */
    using EnqueueHandler = std::function<uint32_t()>;

    /**
     * @brief Callback invoked by wait operation.
     * @param requestID id of request to be used for wait
     * @param timeoutMilliseconds timeout of wait in milliseconds
     * @return Status of subrequest @see RequestStatus
     *
     */
    using WaitHandler = std::function<RequestStatus(uint32_t requestID, int64_t timeoutMilliseconds)>;

    virtual ~Subrequest() = default;

    /**
     * @brief Wait until subrequest will be finished for given timeout.
     * @param timeoutMilliseconds timeout in milliseconds
     * @return status of execution of subrequest @see RequestStatus
     */
    virtual RequestStatus wait(int64_t timeoutMilliseconds) = 0;

    /**
     * @brief Add subrequest to execution queue.
     * @return true in case subrequest was properly enqueued, otherwise return false
     */
    virtual bool enqueue() = 0;

    /**
     * @brief Finalize subrequest and set it status to RequestStatus::kNone
     */
    virtual void cleanup() = 0;

    /**
     * @brief Return true if subrequest is pending, otherwise return false
     */
    virtual bool isPending() const = 0;

    /**
     * @brief Return true if subrequest is aborted, otherwise return false
     */
    virtual bool isAborted() const = 0;

    /**
     * @brief Return true if subrequest is completed, otherwise return false
     */
    virtual bool isCompleted() const = 0;
};

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
