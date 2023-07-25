// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <cstdlib>

#include "backend/dnn_types.hpp"

#define CNN_MAX_POOL_SIZE 6

void CNNFilter32(intel_dnn_component_t* component);
void CNNMaxPool(intel_dnn_component_t* component,
                intel_dnn_number_type_t number_type,
                const bool fused_with_convolution_2d,
                const bool sumPoolingOverRide = false);

void CNN2DFilter32(intel_dnn_component_t* component);
