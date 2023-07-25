// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_batch_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

bool SpaceToBatchKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SPACE_TO_BATCH ||
        o.GetType() != KernelType::SPACE_TO_BATCH) {
        return false;
    }

    const space_to_batch_params& params = static_cast<const space_to_batch_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 6)
        return false;

    return true;
}

CommonDispatchData SpaceToBatchKernelBase::SetDefault(const space_to_batch_params& params, const optional_params&) const {
    const auto& out = params.outputs[0];
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    CommonDispatchData dispatchData;
    if (out.GetLayout() == DataLayout::b_fs_yx_fsv16 && out.Feature().v % 16 == 0) {
        dispatchData.gws = { out.Batch().v, out.Feature().v, out.Y().v * out.X().v };
        dispatchData.lws = { 1, 16, 1 };
    } else {
        dispatchData.gws = { out.Batch().v, out.Feature().v, out.W().v * out.Z().v * out.Y().v * out.X().v };
        dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                       { Tensor::DataChannelName::FEATURE },
                       { Tensor::DataChannelName::X, Tensor::DataChannelName::Y,
                         Tensor::DataChannelName::Z, Tensor::DataChannelName::W }};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    return dispatchData;
}

JitConstants SpaceToBatchKernelBase::GetJitConstants(const space_to_batch_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto makeJitConstForParam = [](JitConstants& jit, const std::string name, const DimTensor<uint32_t>& args, const size_t default_value) {
        jit.AddConstant(MakeJitConstant(name + "_SIZES", args));
        jit.AddConstant(MakeJitConstant(name + "_BATCH", args.b));
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", args.f));
        jit.AddConstant(MakeJitConstant(name + "_Y", args.y));
        jit.AddConstant(MakeJitConstant(name + "_X", args.x));

        if (args.w != 0) {
            jit.AddConstant(MakeJitConstant(name + "_W", args.w));
            jit.AddConstant(MakeJitConstant(name + "_Z", args.z));
        } else if (args.z != 0) {
            jit.AddConstant(MakeJitConstant(name + "_W", default_value));
            jit.AddConstant(MakeJitConstant(name + "_Z", args.z));
        } else {
            jit.AddConstant(MakeJitConstant(name + "_W", default_value));
            jit.AddConstant(MakeJitConstant(name + "_Z", default_value));
        }
    };

    makeJitConstForParam(jit, "BLOCK_SHAPE", params.block_shape, 1);
    makeJitConstForParam(jit, "PADS_BEGIN", params.pads_begin, 0);
    makeJitConstForParam(jit, "PADS_END", params.pads_end, 0);

    return jit;
}

KernelsData SpaceToBatchKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<space_to_batch_params>(params);
    space_to_batch_params& newParams = *static_cast<space_to_batch_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, 1, GetFusedPrimitiveInputsCount(params));

    return { kd };
}
}  // namespace kernel_selector
