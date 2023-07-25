﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// roi_pooling_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct roi_pooling_params : public base_params {
    roi_pooling_params() : base_params(KernelType::ROI_POOLING) {}

    PoolType mode = PoolType::MAX;
    bool position_sensitive = false;
    int pooled_width = 0;
    int pooled_height = 0;
    int spatial_bins_x = 1;
    int spatial_bins_y = 1;
    float spatial_scale = 1.f;
    float trans_std = 1.f;
    bool no_trans = true;
    int part_size = 1;
    int group_size = 1;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        if (position_sensitive) {
            k.EnablePositionSensitivePooling();
        }
        k.EnablePoolType(mode);

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// roi_pooling_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct roi_pooling_optional_params : optional_params {
    roi_pooling_optional_params() : optional_params(KernelType::ROI_POOLING) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ROIPoolingKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ROIPoolingKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~ROIPoolingKernelBase() {}

    using DispatchData = CommonDispatchData;

    KernelsData GetCommonKernelsData(const Params& params, const optional_params& options) const;

protected:
    virtual JitConstants GetJitConstants(const roi_pooling_params& params) const;
};
}  // namespace kernel_selector
