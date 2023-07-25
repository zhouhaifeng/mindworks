// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "gna_device_interface.hpp"

namespace ov {
namespace intel_gna {

class MockGNADevice : public GNADevice {
public:
    MOCK_METHOD(uint32_t, createModel, (Gna2Model&), (const, override));
    MOCK_METHOD(uint32_t, createRequestConfig, (const uint32_t), (const, override));
    MOCK_METHOD(uint32_t, enqueueRequest, (const uint32_t, Gna2AccelerationMode), (override));
    MOCK_METHOD(RequestStatus, waitForRequest, (uint32_t, int64_t), (override));
    MOCK_METHOD(uint32_t, maxLayersCount, (), (const, override));
};

}  // namespace intel_gna
}  // namespace ov
