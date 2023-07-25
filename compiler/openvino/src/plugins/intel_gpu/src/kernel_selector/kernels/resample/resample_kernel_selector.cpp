﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_selector.h"
#include "resample_kernel_ref.h"
#include "resample_kernel_opt.h"
#include "resample_kernel_onnx.h"

namespace kernel_selector {
resample_kernel_selector::resample_kernel_selector() {
    Attach<ResampleKernelRef>();
    Attach<ResampleKernelOpt>();
    Attach<ResampleKernelOnnx>();
}

KernelsData resample_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::RESAMPLE);
}
}  // namespace kernel_selector
