// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "model_wrapper.hpp"

namespace ov {
namespace intel_gna {
namespace request {

class ModelWrapperFactory {
public:
    using ModelInitializer = std::function<void(Gna2Model* gnaModel)>;

    static std::shared_ptr<ModelWrapper> createTrivial();
    static std::shared_ptr<ModelWrapper> createWithNumberOfEmptyOperations(uint32_t number_of_operations);
    static std::shared_ptr<ModelWrapper> createInitialized(ModelInitializer initializer);
};

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
