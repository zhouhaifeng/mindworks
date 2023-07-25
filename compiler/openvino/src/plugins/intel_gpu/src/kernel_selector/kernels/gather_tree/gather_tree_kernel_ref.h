// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gather_tree_kernel_base.h"

namespace kernel_selector {
class GatherTreeKernelRef : public GatherTreeKernelBase {
public:
    GatherTreeKernelRef() : GatherTreeKernelBase("gather_tree_gpu_ref") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
