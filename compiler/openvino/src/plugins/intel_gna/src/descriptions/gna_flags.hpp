// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "gna/gna_config.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {

struct GNAFlags {
    uint8_t num_requests = 1;
    bool compact_mode = true;
    bool exclusive_async_requests = false;
    ov::intel_gna::PWLDesignAlgorithm pwl_design_algorithm = ov::intel_gna::PWLDesignAlgorithm::UNDEFINED;
    bool uniformPwlDesign = false;
    float pwlMaxErrorPercent = 1.0f;
    bool gna_openmp_multithreading = false;
    bool sw_fp32 = false;
    bool performance_counting = false;
    bool input_low_precision = false;
    ov::log::Level log_level = ov::log::Level::NO;
};

}  // namespace intel_gna
}  // namespace ov
