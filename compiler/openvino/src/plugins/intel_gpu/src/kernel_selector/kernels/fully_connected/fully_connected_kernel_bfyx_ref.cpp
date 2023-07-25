﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "fully_connected_kernel_bfyx_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey FullyConnected_bfyx_Ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableAllInputLayout();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableDynamicShapesSupport();
    return k;
}

FullyConnected_bfyx_Ref::DispatchData FullyConnected_bfyx_Ref::SetDefault(const fully_connected_params& params,
                                                                          int) const {
    auto dispatchData = Parent::SetDefault(params);

    std::vector<size_t> global = { params.outputs[0].Feature().v, params.outputs[0].Batch().v, 1 };
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        global = { params.outputs[0].Feature().v, params.outputs[0].Y().v, params.outputs[0].Batch().v };
    }

    dispatchData.gws = global;
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsPriority FullyConnected_bfyx_Ref::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants FullyConnected_bfyx_Ref::GetJitConstants(const fully_connected_params& params,
    const FullyConnectedKernelBase::DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    Datatype accumulator_dt = GetAccumulatorType(params);
    Datatype activation_dt = GetActivationType(params);
    if (params.outputs[0].GetLayout() == DataLayout::bfyx)
        jit.AddConstant(MakeJitConstant("OUTPUT_3D", true));
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = { "b", "ofm", "0", "0" };
        if (params.outputs[0].GetLayout() == DataLayout::bfyx)
            idx_order = { "b", "ofm", "oym", "0" };
        FusedOpsConfiguration conf = { "", idx_order, "dequantized", activation_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }
    return jit;
}

KernelsData FullyConnected_bfyx_Ref::GetKernelsData(const Params& params, const optional_params& options) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(
            params,
            options,
            fc_params.inputs[0].GetLayout(),
            WeightsLayout::oiyx,
            static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

bool FullyConnected_bfyx_Ref::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    // int8 validation
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    // We don't support 4d output
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx && fc_params.outputs[0].X().v > 1)
        return false;

    return true;
}

}  // namespace kernel_selector
