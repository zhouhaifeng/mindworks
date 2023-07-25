// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "request/subrequest.hpp"

namespace ov {
namespace intel_gna {
namespace request {

class MockSubrequest : public Subrequest {
public:
    MOCK_METHOD(RequestStatus, wait, (int64_t), (override));
    MOCK_METHOD(bool, enqueue, (), (override));
    MOCK_METHOD(void, cleanup, (), (override));
    MOCK_METHOD(bool, isPending, (), (const, override));
    MOCK_METHOD(bool, isAborted, (), (const, override));
    MOCK_METHOD(bool, isCompleted, (), (const, override));
};

}  // namespace request
}  // namespace intel_gna
}  // namespace ov
