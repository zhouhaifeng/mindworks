// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file ie_legacy_itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace InferenceEngine {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(IELegacy);
OV_ITT_DOMAIN(IELegacy_LT);
}  // namespace domains
}  // namespace itt
}  // namespace InferenceEngine
