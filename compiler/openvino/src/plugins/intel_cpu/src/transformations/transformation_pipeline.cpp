// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformation_pipeline.h"
#include "defs.hpp"

// Operations
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset10.hpp"
#include <ov_ops/augru_cell.hpp>
#include <ov_ops/augru_sequence.hpp>

// Common transformations
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/broadcast_transition.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/common_optimizations/augru_cell_fusion.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"
#include "transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_roi_align_v3_to_v9.hpp"
#include "transformations/op_conversions/convert_roi_align_v9_to_v3.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_shuffle_channels3.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/op_conversions/convert_space_to_batch.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/op_conversions/detection_output_downgrade.hpp"
#include "transformations/op_conversions/detection_output_upgrade.hpp"
#include "transformations/op_conversions/eye_decomposition.hpp"
#include "transformations/op_conversions/fq_decomposition.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/normalize_l2_decomposition.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"
#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"
#include "transformations/op_conversions/softplus_decomposition.hpp"
#include "transformations/op_conversions/softsign_decomposition.hpp"
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include "transformations/op_conversions/unique_decomposition.hpp"
#include "transformations/op_conversions/convert_topk3.hpp"
#include "transformations/op_conversions/convert_topk11_downgrade.hpp"
#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"
#include "transformations/init_node_info.hpp"
#include "utils/ngraph_transformation.hpp"

// LPT transformations
#include "low_precision/add.hpp"
#include "low_precision/convert_subtract_constant.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/recurrent_cell.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/bias_attribute.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"

// CPU specific transformations
#include "transformations/cpu_opset/convert_to_cpu_specific_opset.hpp"
#include "transformations/snippets/x64/pass/snippets_mark_skipped.hpp"
#include "transformations/cpu_opset/x64/pass/mha_fusion.hpp"
#include "transformations/cpu_opset/x64/pass/convert_to_interaction.hpp"
#include "transformations/cpu_opset/arm/pass/convert_group_conv.hpp"
#include "transformations/cpu_opset/arm/pass/convert_group_conv1d.hpp"
#include "transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.hpp"
#include "transformations/cpu_opset/arm/pass/mish_decomposition.hpp"
#include "transformations/cpu_opset/common/pass/decompose_integer_divide.hpp"
#include "transformations/cpu_opset/common/pass/convert_fq_rnn_to_quantized_rnn.hpp"
#include "transformations/cpu_opset/common/pass/insert_convert_after_extension.hpp"
#include "transformations/cpu_opset/common/pass/move_eltwise_up_data_movement.hpp"
#include "transformations/cpu_opset/common/pass/swap_convert_transpose.hpp"

// Snippets
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/extract_reshapes_from_mha.hpp"

// Misc
#include "nodes/mvn.h"
#include "nodes/normalize.h"
#include "nodes/fake_quantize.h"
#include "nodes/mha.h"
#include "nodes/rnn.h"
#include "dnnl.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>

namespace ov {
namespace intel_cpu {

using const_node_ptr = const std::shared_ptr<const ov::Node>;

bool Transformations::fuse_type_to_convert(const std::shared_ptr<ngraph::Node>& node, const precisions_map& precisions) {
    const auto& from = node->get_output_element_type(0);
    auto it = precisions.find(from);
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto convert = ov::as_type_ptr<ov::opset10::Convert>(node)) {
        // For Convert node, converting precision from floating point to boolean will lead to mathematical
        // error, because here the output precision boolean is replaced by u8. E.g. floating point value 0.01
        // is converted to be 1 for boolean, but 0 for u8. Thus an Abs and Ceil node should be added before the
        // Convert node for this scenario.
        if (convert->input(0).get_element_type().is_real() &&
            convert->get_convert_element_type() == ngraph::element::boolean && to.is_integral_number()) {
            auto abs = std::make_shared<ov::opset10::Abs>(convert->input_value(0).get_node_shared_ptr());
            auto ceil = std::make_shared<ov::opset10::Ceiling>(abs);
            auto new_convert = std::make_shared<ov::opset10::Convert>(ceil, to);
            new_convert->set_friendly_name(convert->get_friendly_name());
            ov::copy_runtime_info(convert, {abs, ceil, new_convert});
            ov::replace_node(convert, new_convert);
            return true;
        } else {
            convert->set_convert_element_type(to);
            return true;
        }
    }
    return false;
}

void Transformations::UpToCpuSpecificOpSet() {
    const bool useLpt = enableLpt &&
        ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(model) &&
        CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(config.debugCaps, Lpt);

    const bool useSnippets = snippetsMode != Config::SnippetsMode::Disable &&
        CPU_DEBUG_CAP_IS_TRANSFORMATION_ENABLED(config.debugCaps, Snippets);

    auto defaultPrecisions = useLpt ? ngraph::pass::low_precision::precision_set::int8_support : std::vector<ov::element::Type>{};
    bool hasINT16orINT32Levels = false;

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part1);
        hasINT16orINT32Levels = ngraph::pass::low_precision::LowPrecision::isFQLevelsPresent(
            model,
            {ngraph::pass::low_precision::levels::int16, ngraph::pass::low_precision::levels::int16_narrow_range,
             ngraph::pass::low_precision::levels::int32, ngraph::pass::low_precision::levels::int32_narrow_range});
        if (hasINT16orINT32Levels) {
            defaultPrecisions = ngraph::pass::low_precision::precision_set::int8_int16_int32_support;
        }
    }

    PreLpt(defaultPrecisions, isLegacyApi);

    if (useLpt)
        Lpt(hasINT16orINT32Levels, defaultPrecisions);

    PostLpt();

    if (useSnippets)
        Snippets();
}

void Transformations::CpuSpecificOpSet(void) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Specific);

    ConvertToCPUSpecificOpset(model);
}

void Transformations::PreLpt(const std::vector<ov::element::Type>& defaultPrecisions, const bool isLegacyApi) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, PreLpt);

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::InitNodeInfo);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::MarkShapeOfSubgraphs);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::KeepConstAndDecompressionForMatMul);

    const bool useLpt = !defaultPrecisions.empty();
    if (useLpt) {
        CPU_REGISTER_PASS_COMMON(manager, ov::pass::MarkDequantizationSubgraph, defaultPrecisions);
    }

    auto get_convert_precisions = []() {
        precisions_map map = {
            {ov::element::i64,     ov::element::i32},
            {ov::element::u64,     ov::element::i32},
            {ov::element::i16,     ov::element::i32},
            {ov::element::u16,     ov::element::i32},
            {ov::element::u32,     ov::element::i32},
            {ov::element::f64,     ov::element::f32},
            {ov::element::f16,     ov::element::f32},
            {ov::element::boolean, ov::element::u8},
            {ov::element::i4,      ov::element::i8},
            {ov::element::u4,      ov::element::u8}
        };

        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
            map.insert({ov::element::bf16, ov::element::f32});

        return map;
    };
    static const auto precisions = get_convert_precisions();
    type_to_fuse_map type_to_fuse = {{ov::opset10::Convert::get_type_info_static(), fuse_type_to_convert}};

    CPU_REGISTER_PASS_COMMON(manager, ov::pass::AUGRUCellFusion);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::BroadcastTransition);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::CommonOptimizations);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::WrapInterpolateIntoTransposes);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::TransposeSinking);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertSequenceToTensorIterator);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertOpSet3ToOpSet2);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertOpSet2ToOpSet1);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::LSTMCellDecomposition);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::GRUCellDecomposition);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::RNNCellDecomposition);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS1ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS3ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS4ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS5ToNMS9);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertNMS9ToNMSIEInternal);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertMulticlassNmsToMulticlassNmsIE);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertMatrixNmsToMatrixNmsIE);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::TransposeMatMul);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part2);
        CPU_REGISTER_PASS_COMMON(manager, ngraph::pass::low_precision::ConvertSubtractConstant, defaultPrecisions);
    }
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    // Common ConvertPrecision pass handles only a limited set of opevino operations to match the list of precisions supported by the plugin.
    // However, if the extension operation produces an output precision that is not natively supported, this may lead to inconsistency during
    // element type propagation. This transformation is called before the ConvertPrecision pass to align the actual precisions with the list of supported ones.
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::InsertConvertAfterExtension);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConvertPrecision, precisions, type_to_fuse);

    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EliminateConvert);
    CPU_REGISTER_PASS_COMMON(manager, SwapConvertTranspose);
    CPU_REGISTER_PASS_X64(manager, ConvertToInteraction);
    CPU_REGISTER_PASS_X64(manager, ConvertInteractionInt8);
    CPU_REGISTER_PASS_ARM(manager, ConvertReduceMultiAxis);
    CPU_REGISTER_PASS_ARM(manager, MishDecomposition);
    CPU_REGISTER_PASS_ARM(manager, ConvertConv1D);
    CPU_REGISTER_PASS_ARM(manager, ConvertGroupConv1D);
    CPU_REGISTER_PASS_ARM(manager, ConvertGroupConvolution);
    // The plugin computes Divide in floating point precision.
    // To preserve correct math for integer division we need to insert explicit Floor operation.
    CPU_REGISTER_PASS_ARM(manager, DecomposeIntegerDivide);
    CPU_REGISTER_PASS_X86(manager, DecomposeIntegerDivide);

    // SpaceToDepth/ DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            return node->input_value(0).get_shape().size() <= 5lu &&
                node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
        },
        ov::pass::ConvertSpaceToDepth, ov::pass::ConvertDepthToSpace);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            const auto & rank = node->input(0).get_partial_shape().rank().get_length();
            return rank == 4lu || rank == 5lu;
        },
        ov::pass::ConvertBatchToSpace, ov::pass::ConvertSpaceToBatch);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string msg;
            return node::RNN::isSupportedOperation(node, msg);
        },
        ov::pass::ConvertRNNSequenceToTensorIterator,
        ov::pass::ConvertGRUSequenceToTensorIterator,
        ov::pass::ConvertLSTMSequenceToTensorIterator);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string msg;
            return node::RNN::isSupportedOperation(node, msg);
        },
        ov::pass::RNNCellDecomposition,
        ov::pass::GRUCellDecomposition,
        ov::pass::LSTMCellDecomposition);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string errorMessage;
            return node::MVN::isSupportedOperation(node, errorMessage);
        },
        ov::pass::MVN6Decomposition);

    CPU_SET_CALLBACK_COMMON(manager,
        [](const_node_ptr &node) -> bool {
            std::string errorMsg;
            return node::NormalizeL2::isSupportedOperation(node, errorMsg);
        },
        ov::pass::NormalizeL2Decomposition);

    CPU_ENABLE_PASS_COMMON(manager, ov::pass::SoftmaxDecomposition);
    CPU_SET_CALLBACK_COMMON(manager,
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
            },
        ov::pass::SoftmaxDecomposition);

    // NMS-alike nodes are always transformed to NMSIEInternal node in case of legacy api, for compatibility.
    // And on the other hand in case of api 2.0, keep them internal dynamic for better performance and functionality.
    auto nmsCallback = [isLegacyApi](const_node_ptr &node) -> bool {
        return isLegacyApi ?  false : true;
    };

    CPU_SET_CALLBACK_COMMON(manager, nmsCallback, ov::pass::ConvertNMS9ToNMSIEInternal);
    CPU_SET_CALLBACK_COMMON(manager, nmsCallback, ov::pass::ConvertMulticlassNmsToMulticlassNmsIE);
    CPU_SET_CALLBACK_COMMON(manager, nmsCallback, ov::pass::ConvertMatrixNmsToMatrixNmsIE);

    // List of enabled/disabled transformations

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::DisableDecompressionConvertConstantFolding);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertCompressedOnlyToLegacy);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::EyeDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertGELU);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertShuffleChannels3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::Gelu7Downgrade);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SoftPlusDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertMod);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertShuffleChannels3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::WeightsDequantizeToFakeQuantize);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SimplifyCTCGreedyDecoderSeqLen);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertGather7ToGather1);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertGather8ToGather7);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertMinimum);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertBroadcastToTiles);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertReduceMeanToPooling);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertReduceMaxToPooling);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertReduceSumToPooling);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SliceToStridedSlice);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertDetectionOutput8ToDetectionOutput1);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertROIAlign9To3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::SoftSignDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::UniqueDecomposition);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertTopK3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::ConvertTopK11ToTopK3);
    CPU_DISABLE_PASS_COMMON(manager, ov::pass::HSwishDecomposition);
    CPU_DISABLE_PASS_X64(manager, ov::pass::HSigmoidDecomposition);

    CPU_DISABLE_PASS_X64(manager, ov::pass::ReduceL1Decomposition);
    CPU_DISABLE_PASS_X64(manager, ov::pass::ReduceL2Decomposition);

    CPU_ENABLE_PASS_COMMON(manager, ov::pass::NormalizeL2Decomposition);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertInterpolate1ToInterpolate4);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertGather1ToGather7);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertDetectionOutput1ToDetectionOutput8);
    CPU_ENABLE_PASS_COMMON(manager, ov::pass::ConvertROIAlign3To9);

    if (useLpt) {
        CPU_LPT_SCOPE(LowPrecisionTransformations_Part3);
        CPU_SET_CALLBACK_COMMON(manager,
            [](const_node_ptr &node) -> bool {
                std::string errMsg;
                return !node::FakeQuantize::isSupportedOperation(node, errMsg);
            },
            ov::pass::AddFakeQuantizeFusion,
            ov::pass::MulFakeQuantizeFusion,
            ov::pass::FakeQuantizeMulFusion);

        CPU_SET_CALLBACK_COMMON(manager,
            [&defaultPrecisions](const_node_ptr &node) -> bool {
                return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
            },
            ov::pass::ConvertQuantizeDequantize);
    }

    /* In some cases, during the transformation pipeline, some MatMul nodes can be transformed into other nodes. For example, they can become part of
       AUGRUCell node (see AUGRUCellFusion pass). In such cases, some constant paths will be unfolded, which can lead to crashes in the plugin. To avoid this,
       we re-mark decompression converts again and finally do CF for those constant paths that are not inputs to MatMul node */
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EnableDecompressionConvertConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::KeepConstAndDecompressionForMatMul);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);

    manager.run_passes(model);
}

void Transformations::Lpt(const bool hasINT16orINT32Levels, const std::vector<ov::element::Type>& defaultPrecisions) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Lpt);

    using namespace ngraph::pass::low_precision;
    CPU_LPT_SCOPE(LowPrecisionTransformations_Part4);
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "LowPrecisionTransformations");
    //Only enable conv/group conv signed input on AMX platform.
    std::vector<ov::element::Type> input0LowPrecisionList;
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
        input0LowPrecisionList = {ov::element::u8, ov::element::i8};
    } else {
        input0LowPrecisionList = {ov::element::u8};
    }

    auto supportedPrecisions = std::vector<PrecisionsRestriction>({
            PrecisionsRestriction::create<ov::opset1::Convolution>({
                    {{0}, input0LowPrecisionList},
                    {{1}, {ov::element::i8}},
                }),
            PrecisionsRestriction::create<ov::opset1::ConvolutionBackpropData>({
                    {{0}, {ov::element::u8, ov::element::i8}},
                    {{1}, {ov::element::i8}}
                }),
            PrecisionsRestriction::create<ov::opset1::GroupConvolution>([input0LowPrecisionList](const std::shared_ptr<ov::Node>& node){
                const auto& input_partial_shape = node->get_input_partial_shape(0);
                const auto& rank = input_partial_shape.rank();
                if (rank.is_static() && (rank.get_length() == 5)) {
                    return PrecisionsRestriction::PrecisionsByPorts{
                        {{0}, {ov::element::u8, ov::element::i8}},
                        {{1}, {ov::element::i8}}};
                }

                return PrecisionsRestriction::PrecisionsByPorts{
                    {{0}, input0LowPrecisionList},
                    {{1}, {ov::element::i8}}
                };
                }),
            PrecisionsRestriction::create<ov::opset1::Multiply>({
                    {{0}, {ov::element::u8}},
                    {{1}, {ov::element::i8}},
                }),
            PrecisionsRestriction::create<ov::opset1::MatMul>({
                    {{0}, {ov::element::u8, ov::element::i8}},
                    {{1}, {ov::element::i8}}
                }),
            PrecisionsRestriction::create<ov::opset5::LSTMSequence>({
                    {{0, 1}, {ov::element::u8}}
                }),
            PrecisionsRestriction::create<ov::opset6::GRUSequence>({
                    {{0, 1}, {ov::element::u8}}
                }),
        });

    auto quantizationRestrictions = std::vector<QuantizationGranularityRestriction>({
            QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0}),
            QuantizationGranularityRestriction::create<ov::opset1::ConvolutionBackpropData>({0})
        });

    // for GNA networks reference execution
    bool updatePrecision = true;
    if (hasINT16orINT32Levels) {
        updatePrecision = false;
        supportedPrecisions = std::vector<PrecisionsRestriction>({});
    }

    ov::pass::Manager lptManager;
    CPU_REGISTER_PASS_COMMON(lptManager, ngraph::pass::low_precision::LowPrecision,
        supportedPrecisions,
        quantizationRestrictions,
        LayerTransformation::Params(updatePrecision, ov::element::f32, defaultPrecisions));
    CPU_SET_CALLBACK_COMMON(lptManager,
        [](const_node_ptr& node) -> bool {
            if (const auto mulitply = std::dynamic_pointer_cast<const ov::opset1::Multiply>(node)) {
                return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
            }
            return false;
        },
        ngraph::pass::low_precision::MarkupPrecisions);
    CPU_SET_CALLBACK_COMMON(lptManager,
        [&defaultPrecisions](const_node_ptr& node) -> bool {
            return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) ||
                WeightableLayerTransformation::isAsymmetricOnWeights(node, defaultPrecisions);
        },
        ngraph::pass::low_precision::ConvolutionBackpropDataTransformation);

    lptManager.get_pass_config()->set_callback<ngraph::pass::low_precision::AddTransformation>(
        [](const_node_ptr& node) -> bool {
            return ov::marked_as_bias(node);
        });

    CPU_DISABLE_PASS_ARM(lptManager, ngraph::pass::low_precision::RecurrentCellTransformation);
    CPU_DISABLE_PASS_COMMON(lptManager, ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation);

    lptManager.run_passes(model);
}

void Transformations::PostLpt() {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, PostLpt);

    ov::pass::Manager postLPTPassManager;
    postLPTPassManager.set_per_pass_validation(false);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::UnrollTensorIterator);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::ReshapePRelu);
    CPU_SET_CALLBACK_COMMON(postLPTPassManager,
        [](const_node_ptr &node) -> bool {
            // UnrollTI transformation is disabled by default, is turned on by LowLatency transformation
            return node->get_rt_info().count("UNROLL_TI") == 0;
        },
        ov::pass::UnrollTensorIterator);
    CPU_REGISTER_PASS_COMMON(postLPTPassManager, MoveEltwiseUpThroughDataMov);
    CPU_SET_CALLBACK_COMMON(postLPTPassManager,
        [](const std::shared_ptr<const ov::Node>& node) -> bool {
            if (node->get_input_size() >= 2) {
                return node->get_input_element_type(1) == ov::element::i8 || node->get_input_element_type(1) == ov::element::u8;
            }
            return false;
        },
        MoveEltwiseUpThroughDataMov);

    CPU_REGISTER_PASS_COMMON(postLPTPassManager, ov::pass::ConstantFolding);

    // Snippets may brake MHA patterns so the fusion has to performed before
    CPU_REGISTER_PASS_X64(postLPTPassManager, MHAFusion);
    CPU_REGISTER_PASS_X64(postLPTPassManager, FuseFQtoInteraction);

    CPU_SET_CALLBACK_X64(postLPTPassManager,
        ([this](const std::shared_ptr<const ov::Node>& n) -> bool {
            std::string errorMessage;

            if (!node::MHA::isSupportedOperation(n, errorMessage))
                return true;

            // Implementation calls AMX BF16 brgemm only for tensors with K and N aligned on 2, otherwise fallbacks on vector impl
            // Vector madd BF16 instruction on SPR has reduced performance on HW level, which results in overall perf degradation
            size_t bf16Factor = 2;
            if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
                (n->get_input_element_type(0) == element::bf16 || (n->get_input_element_type(0) == element::f32 && inferencePrecision == ov::element::bf16)) &&
                (n->get_input_shape(0)[3] % bf16Factor != 0 || n->get_input_shape(1)[1] % bf16Factor != 0 || n->get_input_shape(3)[3] % bf16Factor != 0)) {
                return true;
            }

            return false;
        }),
        MHAFloatFusion, MHAFloatFusion2, MHAQuantFusion, MHAQuantFusion2);

    // Float MHA is supported by snippets now
    if (inferencePrecision == ov::element::f32) {
        CPU_DISABLE_PASS_X64(postLPTPassManager, MHAFloatFusion);
        CPU_DISABLE_PASS_X64(postLPTPassManager, MHAFloatFusion2);
    }

    // Execute before snippets. Otherwise FQ will be converted to Subgraph
    CPU_REGISTER_PASS_X64(postLPTPassManager, ConvertFqRnnToQuantizedRnn);
    postLPTPassManager.run_passes(model);
}

void Transformations::MainSnippets(void) {
    if (snippetsMode == Config::SnippetsMode::Disable ||
        !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) // snippets are implemented only for relevant platforms (avx2+ extensions)
        return;

    ov::snippets::pass::SnippetsTokenization::Config tokenization_config;
    // At the moment Snippets supports Transposes in MHA pattern only in FP32 case since
    //      - ConvertSaturation[BF16->FP32] will be inserted after Parameters and before Transposes in canonicalization stage
    //      - ConvertSaturation[FP32->BF16] will be inserted after Transposes and before Brgemm in precision propagation stage
    // Because of that Transposes won't be fused into Brgemm
    // TODO [111813]: Need to update this pipeline to avoid Converts between Transposes and Brgemm on inputs
    tokenization_config.mha_token_enable_transpose = (inferencePrecision == ov::element::f32);
    tokenization_config.minimal_concurrency = parallel_get_num_threads();
    // The optimization "SplitDimensionM" depends on target machine (thread count).
    // To avoid uncontrolled behavior in tests, we disabled the optimization when there is Config::SnippetsMode::IgnoreCallback
    tokenization_config.split_m_dimension = snippetsMode != Config::SnippetsMode::IgnoreCallback;

    ngraph::pass::Manager snippetsManager;
    snippetsManager.set_per_pass_validation(false);
    if (snippetsMode != Config::SnippetsMode::IgnoreCallback)
        CPU_REGISTER_PASS_X64(snippetsManager, SnippetsMarkSkipped, inferencePrecision != ov::element::f32);
    CPU_REGISTER_PASS_X64(snippetsManager, snippets::pass::SnippetsTokenization, tokenization_config);

    // Tokenize MHA in quantized model or with BF16 only in tests.
    // TODO [106921]: Please enable the tokenization when the ticket 106921 with blocking support for BRGEMM will be implemented
    const bool onlyFloatSupported = snippetsMode != Config::SnippetsMode::IgnoreCallback;
    const bool isMHASupported =
            IMPLICATION(inferencePrecision != ov::element::f32, !onlyFloatSupported) &&
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);  // MHA has BRGEMM that is supported only on AVX512 platforms
    if (!isMHASupported) {
        CPU_DISABLE_PASS_X64(snippetsManager, snippets::pass::TokenizeMHASnippets);
        CPU_DISABLE_PASS_X64(snippetsManager, snippets::pass::ExtractReshapesFromMHA);
    }

    if (snippetsMode != Config::SnippetsMode::IgnoreCallback) {
#if defined(OPENVINO_ARCH_X86_64)
        auto is_supported_matmul = [onlyFloatSupported](const std::shared_ptr<const ov::Node>& n) {
            const auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(n);
            if (!matmul)
                return false;
            if (matmul->get_input_element_type(1) == ov::element::i8)
                return !onlyFloatSupported && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_vnni);
            if (matmul->get_input_element_type(0) == ov::element::bf16 &&
                matmul->get_input_element_type(1) == ov::element::bf16)
                return !onlyFloatSupported && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16);
            return true;
        };
        auto is_unsupported_parallel_work_amount = [&](const std::shared_ptr<const ov::Node>& n, const ov::Shape& shape) {
            const auto parallel_work_amount = std::accumulate(shape.rbegin() + 2, shape.rend(), 1, std::multiplies<size_t>());
            // Heuristic values:
            //    parallelism work amount - not enough work amount for parallelism
            // TODO: The heuristic will be removed after parallelism support on JIT level
            const auto needed_num_of_threads = 12lu;
            const auto is_unsupported_parallel_work_amount =
                parallel_get_num_threads() / 2 > parallel_work_amount &&
                static_cast<size_t>(parallel_work_amount) < needed_num_of_threads &&
                !ov::snippets::pass::CommonOptimizations::CanOptimizeParallelWA(n, tokenization_config.minimal_concurrency);
            return is_unsupported_parallel_work_amount;
        };
#endif // OPENVINO_ARCH_X86_64
        CPU_SET_CALLBACK_X64(snippetsManager, [&](const std::shared_ptr<const ov::Node>& n) -> bool {
            // Tranformation callback is called on MatMul0
            if (!is_supported_matmul(n))
                return true;
            // Search for MatMul1
            auto child = n->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            while (!ov::is_type<const ov::op::v0::MatMul>(child)) {
                child = child->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            }
            if (!is_supported_matmul(child))
                return true;

            const auto& shape = child->get_input_shape(0);
            return is_unsupported_parallel_work_amount(n, shape);
        }, snippets::pass::TokenizeMHASnippets);
        CPU_SET_CALLBACK_X64(snippetsManager, [&](const std::shared_ptr<const ov::Node>& n) -> bool {
            return !is_supported_matmul(n) || is_unsupported_parallel_work_amount(n, n->get_output_shape(0));
        }, snippets::pass::ExtractReshapesFromMHA);
        CPU_SET_CALLBACK_X64(snippetsManager,
            [](const std::shared_ptr<const ov::Node>& n) -> bool {
                // CPU Plugin support Swish in Subgraph via conversion to SwichCPU which assumes second input to be constant
                const bool is_unsupported_swish =
                        ov::is_type<const ov::op::v4::Swish>(n) && n->inputs().size() > 1 &&
                        !ov::is_type<const ov::op::v0::Constant>(n->get_input_node_shared_ptr(1));
                // todo: general tokenization flow is not currently supported for these operations.
                //  they can be tokenized only as a part of complex patterns
                const bool is_disabled_tokenization = (ov::is_type<const ov::op::v1::Softmax>(n) ||
                                                       ov::is_type<const ov::op::v8::Softmax>(n) ||
                                                       ov::is_type<const ov::op::v0::MatMul>(n) ||
                                                       ov::is_type<const ov::op::v1::Transpose>(n) ||
                                                       ov::is_type<const ov::op::v1::Broadcast>(n) ||
                                                       ov::is_type<const ov::op::v3::Broadcast>(n));
                const auto& inputs = n->inputs();
                // todo: clarify whether we can evaluate snippets on const paths
                const bool has_only_const_inputs = std::all_of(inputs.begin(), inputs.end(),
                                                               [](const ov::Input<const ov::Node>& in) {
                                                                   return ov::is_type<ov::op::v0::Constant>(
                                                                           in.get_source_output().get_node_shared_ptr());
                                                               });
                // todo: clarify whether we can evaluate snippets on inputs with larger ranks
                auto rank_is_too_large = [](const ov::descriptor::Tensor& t) {
                    // callback is called has_supported_in_out(), so it's safe to assume that the shapes are static
                    return t.get_partial_shape().rank().get_length() > 6;
                };
                const bool bad_input_rank = std::any_of(inputs.begin(), inputs.end(),
                                                        [&](const ov::Input<const ov::Node>& in) {
                                                            return rank_is_too_large(in.get_tensor());
                                                        });
                const auto& outputs = n->outputs();
                const bool bad_output_rank = std::any_of(outputs.begin(), outputs.end(),
                                                        [&](const ov::Output<const ov::Node>& out) {
                                                            return rank_is_too_large(out.get_tensor());
                                                        });
                return has_only_const_inputs || bad_input_rank || bad_output_rank || is_unsupported_swish ||
                    is_disabled_tokenization;
            },
            snippets::pass::TokenizeSnippets);
    }
    snippetsManager.run_passes(model);
}

void Transformations::PostSnippets(void) {
    ov::pass::Manager postSnippetsManager;
    postSnippetsManager.set_per_pass_validation(false);
    CPU_REGISTER_PASS_COMMON(postSnippetsManager, ov::pass::FakeQuantizeDecomposition);
    CPU_SET_CALLBACK_COMMON(postSnippetsManager,
        [](const_node_ptr& node) -> bool {
            std::string errMsg;
            return node::FakeQuantize::isSupportedOperation(node, errMsg);
        },
        ov::pass::FakeQuantizeDecomposition);
    CPU_REGISTER_PASS_COMMON(postSnippetsManager, ov::pass::ConstantFolding);
    postSnippetsManager.run_passes(model);
}

void Transformations::Snippets(void) {
    CPU_DEBUG_CAP_TRANSFORMATION_SCOPE(this, Snippets);

    MainSnippets();
    PostSnippets();
}

}   // namespace intel_cpu
}   // namespace ov
