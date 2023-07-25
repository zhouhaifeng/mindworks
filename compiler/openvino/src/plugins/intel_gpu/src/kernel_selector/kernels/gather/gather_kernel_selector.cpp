// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_kernel_selector.h"
#include "gather_kernel_ref.h"

namespace kernel_selector {

gather_kernel_selector::gather_kernel_selector() { Attach<GatherKernelRef>(); }

KernelsData gather_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::GATHER);
}
}  // namespace kernel_selector
