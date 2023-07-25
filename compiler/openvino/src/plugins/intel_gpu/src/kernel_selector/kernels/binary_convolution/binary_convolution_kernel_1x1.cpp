﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "binary_convolution_kernel_1x1.h"
#include <string>
#include <activation/activation_kernel_base.h>
#include <eltwise/eltwise_kernel_base.h>

namespace kernel_selector {

static const int sub_group_size = 16;
static const int ic_pack_size = 32;
static const int xy_block_size = 16;

ParamsKey BinaryConvolutionKernel1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey BinaryConvolutionKernel1x1::get_required_device_features_key(const Params& params, const optional_params& /*options*/) const {
    DeviceFeaturesKey k;
    k.requires_subgroup_shuffle();
    k.requires_blocked_read_write();

    return k;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernel1x1::SetDefault(const binary_convolution_params& params, int) const {
    DispatchData dispatchData = BinaryConvolutionKernelBase::SetDefault(params);

    const auto& out = params.outputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = Align(x * y, sub_group_size);
    dispatchData.gws[1] = CeilDiv(f, 2 * sub_group_size);  // 1 WI calcs 32 OC
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = sub_group_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority BinaryConvolutionKernel1x1::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

bool BinaryConvolutionKernel1x1::Validate(const Params& p, const optional_params& o) const {
    if (!BinaryConvolutionKernelBase::Validate(p, o) || !ConvolutionBinaryCheckInput(p, o))
        return false;

    const auto& params = static_cast<const binary_convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    const bool bOutputSizes = output.X().v != input.X().v || output.Y().v != input.Y().v;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;
    const bool bGroups = params.groups > 1;

    if (bOutputSizes || bFilterSize || bStride || bGroups)
        return false;

    return true;
}

JitConstants BinaryConvolutionKernel1x1::GetJitConstants(const binary_convolution_params& params,
                                                         const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_NUM_PACKED", CeilDiv(params.inputs[0].Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_NUM_PACKED", CeilDiv(params.outputs[0].Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", params.outputs[0].X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("XY_BLOCK_SIZE", xy_block_size));
    if (params.inputs[0].Feature().v % ic_pack_size) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_IC", params.inputs[0].Feature().v % ic_pack_size));
        jit.AddConstant(MakeJitConstant("FILTER_MASK",
                                        (0xFFFFFFFF >> (ic_pack_size - params.inputs[0].Feature().v % ic_pack_size))));
    }

    if (params.outputs[0].GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_PACKED_OUTPUT", 1));
    }

    return jit;
}

JitConstants BinaryConvolutionKernel1x1::GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                                        const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    auto input_dt = GetUnitType(params);
    FusedOpsConfiguration conf = {"", {"b", "(f_block*16 + oc)", "y", "x"}, "res", input_dt, 1 };
    jit.Merge(MakeFusedOpsDeclsJitConstants(params, {conf}));

    size_t op_id = 0;
    std::string input_decls = "";
    std::string eltwise_fused_ops = "";
    std::string prepare_data = "";
    for (auto& fused_dep : params.fused_ops) {
        auto fused_dep_codegen = FusedOpsCodeGenerator(fused_dep);
        auto get_aligned_load2 = [&](std::string ptr, std::string byte_offset) -> std::string {
            if (fused_dep.tensors[0].GetDType() == Datatype::F32)
                return "(_sub_group_block_read2((const __global uint*)(" + ptr + ") + (" + byte_offset + ")))";
            else
                return "(_sub_group_block_read_us2((const __global ushort*)(" + ptr + ") + (" + byte_offset +
                       ")))";
        };

        auto get_shuffle = [&](std::string var, std::string lid) -> std::string {
            return "(_sub_group_shuffle(" + var + ", " + lid + "))";
        };

        std::string data_type = fused_dep_codegen.GetInputTypeName(0, 1);
        std::string vec_data_type = fused_dep_codegen.GetInputTypeName(0, 2);
        std::string sc = "sc" + toCodeString(op_id);
        std::string sh = "sh" + toCodeString(op_id);
        std::string e_add = "e_add" + toCodeString(op_id);
        std::string e_mul = "e_mul" + toCodeString(op_id);

        switch (fused_dep.GetType()) {
            case KernelType::QUANTIZE: {
                std::string var_name_in = fused_dep_codegen.GetInputVarName(0);
                std::string var_name_out = fused_dep_codegen.GetInputVarName(3);
                std::string cast_type_vec = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float2" : "as_half2";
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float" : "as_half";

                prepare_data += "\\\n\tint packed_res = 0;";
                if (fused_dep.tensors[0].Feature().v == params.outputs[0].Feature().v) {
                    prepare_data += "\\\n\t" + vec_data_type + " " + var_name_in + " = " + cast_type_vec +
                                    get_aligned_load2(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops += "\\\n\t" + data_type + " thresh = (oc < 16) ? " + get_shuffle(var_name_in + ".s0", "oc") +
                                         " : " + get_shuffle(var_name_in + ".s1", "oc") + ";";
                } else {
                    prepare_data += "\\\n\t" + data_type + " " + var_name_in + " = " + cast_type +
                                 + "(" + fused_dep_codegen.GetInputPtrName(0) + "[0]);";
                    eltwise_fused_ops += "\\\n\t" + data_type + " thresh = " + var_name_in + ";";
                }


                if (fused_dep.tensors[2].Feature().v == params.outputs[0].Feature().v) {
                    // Per-channel output value
                    prepare_data += "\\\n\t" + vec_data_type + " " + var_name_out + " = " + cast_type_vec +
                                    get_aligned_load2(fused_dep_codegen.GetInputPtrName(3), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops +="\\\n\t" + data_type + " out_val = (oc < 16) ? " + get_shuffle(var_name_out + ".s0", "oc") +
                                         " : " + get_shuffle(var_name_out + ".s1", "oc") + ";";
                } else {
                    // Per-tensor output value
                    prepare_data += "\\\n\t" + data_type + " " + var_name_out + " = " + cast_type +
                                    + "(" + fused_dep_codegen.GetInputPtrName(3) + "[0]);";
                    eltwise_fused_ops += "\\\n\t" + data_type + " out_val = " + var_name_out + ";";
                }
                eltwise_fused_ops += "\\\n\tif (out_val == 1) ";
                eltwise_fused_ops += "\\\n\t\tpacked_res |= (res > thresh) << oc;";
                eltwise_fused_ops += "\\\n\telse ";
                eltwise_fused_ops += "\\\n\t\tpacked_res |= (res <= thresh) << oc;";

                break;
            }

            case KernelType::ACTIVATION: {
                auto p = fused_dep.GetOpParams<activation_fuse_params>();
                base_activation_params activation = p->param;
                if (activation.function != ActivationFunction::NONE) {
                    auto suffix = "_FUSED_OP" + toCodeString(op_id);

                    jit.Merge(MakeActivationJitConstants(activation, fused_dep.output_tensor.GetDType(), suffix));
                    eltwise_fused_ops += "\\\n\tres = ACTIVATION" + suffix + "((OUTPUT_TYPE)res, ACTIVATION_PARAMS" + suffix + ");";
                }

                break;
            }

            case KernelType::ELTWISE: {
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float2" : "as_half2";
                std::string var_name = fused_dep_codegen.GetInputVarName(0);
                prepare_data += "\\\n\t" + vec_data_type + " " + var_name + " = " + cast_type +
                                get_aligned_load2(fused_dep_codegen.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";

                auto eltwise_p = std::dynamic_pointer_cast<eltwise_fuse_params>(fused_dep.op_params);

                if (eltwise_p->mode == EltwiseMode::ADD) {
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + e_add + " = (oc < 16) ? " +
                                         get_shuffle(var_name + ".s0", "oc") + " : " + get_shuffle(var_name + ".s1", "oc") + ";";
                    eltwise_fused_ops += "\\\n\tres = res+" + e_add + ";";
                } else if (eltwise_p->mode == EltwiseMode::MUL) {
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + e_mul + " = (oc < 16) ? " +
                                         get_shuffle(var_name + ".s0", "oc") + " : " + get_shuffle(var_name + ".s1", "oc") + ";";
                    eltwise_fused_ops += "\\\n\tres = res*" + e_mul + ";";
                } else {
                    throw std::invalid_argument("Not supported eltwise fusing op in binary_convolution_1x1 kernel: " + params.layerID);
                }

                break;
            }

            default:
                throw std::invalid_argument("Invalid fused op in binary_convolution_1x1 kernel: " + params.layerID);
        }

        op_id++;
    }
    jit.AddConstant(MakeJitConstant("DO_ELTWISE_FUSED_OPS", eltwise_fused_ops));
    jit.AddConstant(MakeJitConstant("FUSED_OPS_PREPARE_DATA", prepare_data));
    jit.AddConstant(MakeJitConstant("CUSTOM_FUSED_OPS", true));

    return jit;
}

KernelsData BinaryConvolutionKernel1x1::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
