// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "binary_convolution_params.h"
#include <sstream>
#include <string>

namespace kernel_selector {
std::string binary_convolution_params::to_string() const {
    std::stringstream s;

    s << base_params::to_string() << "_";
    s << filterSize.x << "_" << filterSize.y << "_";
    s << stride.x << "_" << stride.y << "_";
    s << dilation.x << "_" << dilation.y << "_";
    s << padding.x << "_" << padding.y << "_";
    s << 1;
    s << groups;

    return s.str();
}

std::string binary_convolution_params::to_cache_string_v2() const {
    std::stringstream s;

    s << weight_bias_params::to_cache_string_v2() << ";";
    s << filterSize.x << "_" << filterSize.y << "_" << filterSize.z << ";";
    s << stride.x << "_" << stride.y << "_" << stride.z << ";";
    s << dilation.x << "_" << dilation.y << "_" << dilation.z << ";";
    s << padding.x << "_" << padding.y << "_" << padding.z << ";";
    s << 1 << ";";
    s << groups;

    return s.str();
}

ParamsKey binary_convolution_params::GetParamsKey() const {
    ParamsKey k = weight_bias_params::GetParamsKey();

    if (dilation.x != 1 ||
        dilation.y != 1) {
        k.EnableDilation();
    }

    if (groups > 1) {
        k.EnableGroupedConvolution();
    }

    return k;
}
}  // namespace kernel_selector
