// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-inference-api.h>

#include <memory>

#include "worker.hpp"

namespace ov {
namespace intel_gna {

class GNADevice;

namespace backend {
class AMIntelDNN;
}

namespace request {

class ModelWrapper;
class Subrequest;

class WorkerFactory {
public:
    WorkerFactory() = delete;

    static std::shared_ptr<Worker> createWorker(std::shared_ptr<ModelWrapper> model,
                                                std::shared_ptr<GNADevice> device,
                                                const Gna2AccelerationMode accelerationMode);
    static std::shared_ptr<Worker> createWorkerFP32(std::shared_ptr<ModelWrapper> model,
                                                    std::shared_ptr<backend::AMIntelDNN> dnn);
    static std::shared_ptr<Worker> createWorkerTrivialTopology(std::shared_ptr<ModelWrapper> model);

    static std::vector<std::shared_ptr<Subrequest>> createModelSubrequests(std::shared_ptr<ModelWrapper> model,
                                                                           std::shared_ptr<GNADevice> device,
                                                                           const Gna2AccelerationMode accelerationMode);
    static std::vector<std::shared_ptr<Subrequest>> createModelSubrequestsFP32(
        std::shared_ptr<backend::AMIntelDNN> dnn);
    static std::vector<std::shared_ptr<Subrequest>> createModelSubrequestsTrivial();

private:
    static constexpr const uint32_t kFakeRequestID{1};
};

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
