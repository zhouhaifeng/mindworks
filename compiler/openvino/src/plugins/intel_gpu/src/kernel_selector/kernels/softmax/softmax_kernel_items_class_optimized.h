﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_items_class_kernel_base.h"

namespace kernel_selector {
class SoftmaxKerneItemsClassOptimized : public SoftmaxItemsClassKernelBase {
public:
    using Parent = SoftmaxItemsClassKernelBase;
    SoftmaxKerneItemsClassOptimized() : Parent("softmax_gpu_items_class_optimized") {}
    virtual ~SoftmaxKerneItemsClassOptimized() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params, const optional_params& /*options*/) const override;

protected:
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    DispatchData SetDefault(const softmax_params& params) const override;
};
}  // namespace kernel_selector
