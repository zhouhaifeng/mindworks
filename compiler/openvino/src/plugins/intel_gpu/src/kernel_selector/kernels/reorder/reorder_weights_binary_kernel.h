﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsBinaryKernel : public ReorderKernelBase {
public:
    ReorderWeightsBinaryKernel() : ReorderKernelBase("reorder_weights_binary") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;

protected:
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
