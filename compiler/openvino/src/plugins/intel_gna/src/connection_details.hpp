// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

namespace ov {
namespace intel_gna {

struct ConnectionDetails {
    InferenceEngine::CNNLayerPtr input;
    bool needTransposeWeights = false;
    InferenceEngine::CNNLayerPtr permute;
    ConnectionDetails(InferenceEngine::CNNLayerPtr input,
                      bool bTranspose = false,
                      InferenceEngine::CNNLayerPtr permute = nullptr)
        : input(input),
          needTransposeWeights(bTranspose),
          permute(permute) {}
};

}  // namespace intel_gna
}  // namespace ov
