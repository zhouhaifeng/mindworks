﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_kernel_selector.h"
#include "select_kernel_ref.h"

namespace kernel_selector {
select_kernel_selector::select_kernel_selector() { Attach<SelectKernelRef>(); }

KernelsData select_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::SELECT);
}
}  // namespace kernel_selector
