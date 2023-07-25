// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "worker.hpp"

namespace ov {
namespace intel_gna {
namespace request {

class ModelWrapper;
class Subrequest;

/**
 * @class Implementation of @see Worker interface.
 */
class WorkerImpl : public Worker {
public:
    /**
     * @brief Contrcut {WorkerImpl} object
     * @param model potiner to represented model
     * @param modelSubrequests subrequests to enqueued for given model
     * @throw Exception when model is nullptr,modelSubrequests is empty or any element in modelSubrequests is nullptr
     */
    explicit WorkerImpl(std::shared_ptr<ModelWrapper> model, std::vector<std::shared_ptr<Subrequest>> modelSubrequests);

    WorkerImpl(const WorkerImpl&) = delete;
    WorkerImpl(WorkerImpl&&) = delete;
    WorkerImpl& operator=(const WorkerImpl&) = delete;
    WorkerImpl& operator=(WorkerImpl&&) = delete;

    /**
     * @brief Destroy {WorkerImpl} object
     */
    ~WorkerImpl() override = default;

    /**
     * @see Worker::model()
     */
    const Gna2Model* model() const override;

    /**
     * @see Worker::model()
     */
    Gna2Model* model() override;

    /**
     * @see Worker::enqueueRequest()
     */
    bool enqueueRequest() override;

    /**
     * @see Worker::wait()
     */
    RequestStatus wait(int64_t timeoutMilliseconds) override;

    /**
     * @see Worker::isFree()
     */
    bool isFree() const override;

    /**
     * @see Worker::representingIndex()
     */
    uint32_t representingIndex() const override;

    /**
     * @see Worker::setRepresentingIndex()
     */
    void setRepresentingIndex(uint32_t index) override;

    /**
     * @see Worker::result()
     */
    InferenceEngine::BlobMap& result() override;

    /**
     * @see Worker::setResult()
     */
    void setResult(const InferenceEngine::BlobMap& result) override;

    /**
     * @see Worker::setResult()
     */
    void setResult(InferenceEngine::BlobMap&& result) override;

private:
    void cleanup_subrequests();

    uint32_t representingIndex_{0};
    std::shared_ptr<ModelWrapper> fullModel_;
    std::vector<std::shared_ptr<Subrequest>> modelSubrequests_;
    InferenceEngine::BlobMap requestResult_;
};

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
