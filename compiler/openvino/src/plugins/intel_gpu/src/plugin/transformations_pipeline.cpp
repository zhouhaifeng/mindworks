// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <tuple>
#include <cctype>
#include <memory>

#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/plugin/legacy_api_helper.hpp"

#include <ie_ngraph_utils.hpp>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "transformations/einsum_decomposition.hpp"
#include "transformations/convert_pooling_to_reduce.hpp"
#include "transformations/decompose_reduce_for_false_keepdims.hpp"
#include "transformations/convert_shapeof.hpp"

#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>

#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include "transformations/resolve_names_collisions.hpp"

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include <transformations/common_optimizations/wrap_interpolate_into_transposes.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/softmax_fusion.hpp>
#include <transformations/common_optimizations/broadcast_transition.hpp>

#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_broadcast3.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/softplus_decomposition.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/convert_reduce_to_reshape.hpp>
#include <transformations/op_conversions/convert_shuffle_channels3.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/hsigmoid_decomposition.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include <transformations/op_conversions/mvn6_decomposition.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_9.hpp>
#include <transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_gather_downgrade.hpp>
#include <transformations/op_conversions/convert_gather_0d.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/op_conversions/convert_gp9_to_gp_ie_internal.hpp>
#include <transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/op_conversions/convert_softmax_downgrade.hpp>
#include <transformations/op_conversions/convert_prior_box_v8_to_v0.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/op_conversions/convert_shapeof3.hpp>
#include <transformations/op_conversions/convert_topk11_downgrade.hpp>

#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <low_precision/pull_reshape_through_dequantization.hpp>
#include <low_precision/pull_transpose_through_dequantization.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/convolution_backprop_data.hpp>
#include <low_precision/group_convolution.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/mat_mul.hpp>
#include <low_precision/multiply_to_group_convolution.hpp>
#include <low_precision/strided_slice.hpp>
#include <low_precision/network_helper.hpp>
#include "transformations/op_conversions/eye_decomposition.hpp"
#include <low_precision/recurrent_cell.hpp>

#include "intel_gpu/runtime/itt.hpp"

namespace {
template<typename T>
static bool disableReduceDecomposition(const std::shared_ptr<const ngraph::Node> node) {
    if (auto op = std::dynamic_pointer_cast<const T>(node)) {
        if (op->input(0).get_partial_shape()[0].is_static()) {
            bool fp16_batch_not_1 = op->get_element_type() == ngraph::element::f16 && op->input(0).get_partial_shape()[0] != 1;
            return !fp16_batch_not_1;
        }
    }
    return false;
}
}  // namespace

namespace ov {
namespace intel_gpu {

void TransformationsPipeline::apply(std::shared_ptr<ov::Model> func) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply");
    using const_node_ptr = const std::shared_ptr<const ngraph::Node>;

    const auto defaultPrecisions = ngraph::pass::low_precision::precision_set::int8_support;
    bool enableInt8;
    bool enable_loop_unrolling = config.get_property(ov::intel_gpu::enable_loop_unrolling);
    {
        ngraph::pass::Manager manager;
        manager.set_per_pass_validation(false);

        enableInt8 = config.get_property(ov::intel_gpu::enable_lp_transformations) && ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(func);
        if (enableInt8) {
            manager.register_pass<ov::pass::MarkDequantizationSubgraph>(
                std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4 });
        }

        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<EinsumDecomposition>();

        precisions_map fp_convert_precision_map = {
                {ov::element::f64, ov::element::f32}
        };

        // call conversion of float types with keep_precision_sensitive_in_fp32 = true
        auto fp_precision_supported = [&](ov::element::Type e) -> bool {
            switch (e) {
                case ov::element::f16: return device_info.supports_fp16;
                case ov::element::f32: return true; // assume that all GPUs support f32 data type
                case ov::element::f64: return device_info.supports_fp64;
                case ov::element::bf16: return false;
                default: return false;
            }
            return false;
        };

        const auto fallback_precision = ov::element::f32;
        std::vector<ov::element::Type> fp_element_types = {
                ov::element::f32,
                ov::element::f16,
                ov::element::bf16
        };

        // Add conversion from FP data types to infer precision if it's specified
        auto infer_precision = config.get_property(ov::hint::inference_precision);
        if (infer_precision != ov::element::undefined) {
            if (!fp_precision_supported(infer_precision))
                infer_precision = fallback_precision;

            for (auto& et : fp_element_types) {
                if (et != infer_precision) {
                    fp_convert_precision_map.insert({et, infer_precision});
                }
            }
        }

        // Add conversion from unsupported FP data types to f32 if we don't have a conversion to something valid already in the list
        for (auto& et : fp_element_types) {
            if (!fp_precision_supported(et)) {
                bool has_valid_conversion = fp_convert_precision_map.count(et) && fp_precision_supported(fp_convert_precision_map[et]);
                if (!has_valid_conversion) {
                    fp_convert_precision_map.insert(std::make_pair(et, fallback_precision));
                }
            }
        }

        type_to_fuse_map empty_fuse_map = {};
        manager.register_pass<ov::pass::Validate>();

        // fuse softmax patterns so that they will not be marked as precision sensitive in ConvertPrecision
        manager.register_pass<ov::pass::SoftmaxFusion>();
        // decompose MVNs that sre not supported in GPU, so the they will be marked as precision sensitive in ConvertPrecision
        manager.register_pass<ov::pass::MVN6Decomposition>();
        manager.register_pass<ov::pass::BroadcastTransition>();

        //  call ConvertPrecision with keep_precision_sensitive_in_fp32 = true
        manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, empty_fuse_map, true);

        manager.register_pass<ov::pass::CommonOptimizations>();

        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
        manager.register_pass<ov::pass::TransposeSinking>();

        if (!enable_loop_unrolling) {
            manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
        }

        manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();
        manager.register_pass<ov::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ov::pass::ConvertOpSet2ToOpSet1>();

        manager.register_pass<ov::pass::LSTMCellDecomposition>();
        manager.register_pass<ov::pass::GRUCellDecomposition>();
        manager.register_pass<ov::pass::RNNCellDecomposition>();

        if (enable_loop_unrolling) {
            manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
        }

        manager.register_pass<ConvertShapeOf1To3>();
        manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS9ToNMSIEInternal>();
        manager.register_pass<ov::pass::ConvertGP9ToGPIEInternal>();
        manager.register_pass<ov::pass::ConvertMatrixNmsToMatrixNmsIE>();
        manager.register_pass<ov::pass::ConvertGather0D>();
        manager.register_pass<ov::pass::ConvertPriorBox8To0, false>();
        manager.register_pass<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>();

        precisions_map int_convert_precision_map {
                {ngraph::element::i64, ngraph::element::i32},
                {ngraph::element::u64, ngraph::element::i32},
                {ngraph::element::u16, ngraph::element::i32},
                {ngraph::element::u32, ngraph::element::i32},
                {ngraph::element::boolean, ngraph::element::u8},
                {ngraph::element::i4, ngraph::element::i8},
                {ngraph::element::u4, ngraph::element::u8},
        };

        manager.register_pass<ngraph::pass::Validate>();
        manager.register_pass<ov::pass::ConvertPrecision>(int_convert_precision_map);

        auto pass_config = manager.get_pass_config();
        pass_config->disable<ov::pass::EyeDecomposition>();

        // disable conversion to legacy and use the new mixed precision
        // in which precision sensitive nodes are kept in FP32
        pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

        // SpaceToDepth/DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
        pass_config->set_callback<ov::pass::ConvertSpaceToDepth,
                                  ov::pass::ConvertDepthToSpace>(
                [](const_node_ptr &node) -> bool {
                    return node->input_value(0).get_partial_shape().size() <= 5lu &&
                        node->input_value(0).get_partial_shape().size() == node->get_output_partial_shape(0).size();
                });

        pass_config->set_callback<ov::pass::ConvertBatchToSpace,
                                  ov::pass::ConvertSpaceToBatch>(
                [](const_node_ptr &node) -> bool {
                    const auto & rank = node->input(0).get_partial_shape().rank().get_length();
                    return rank <= 5;
                });

        // Convert reduce to reshape expected to be optimized out
        manager.register_pass<ov::pass::ConvertReduceToReshape>();

        if (device_info.supports_immad) {
            // oneDNN reduction is used
            pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
            pass_config->disable<ov::pass::ConvertReduceMeanToPooling>();
            pass_config->disable<ov::pass::ConvertReduceMaxToPooling>();
            manager.register_pass<ConvertAvgPoolingToReduce>();
            manager.register_pass<DecomposeReduceForFalseKeepDims>();
        } else {
            pass_config->set_callback<ov::pass::ConvertReduceSumToPooling>(
            [](const_node_ptr &node) -> bool {
                return disableReduceDecomposition<ngraph::opset1::ReduceSum>(node);
            });

            pass_config->set_callback<ov::pass::ConvertReduceMeanToPooling>(
            [](const_node_ptr &node) -> bool {
                return disableReduceDecomposition<ngraph::opset1::ReduceMean>(node);
            });

            pass_config->set_callback<ov::pass::ConvertReduceMaxToPooling>(
            [](const_node_ptr &node) -> bool {
                return disableReduceDecomposition<ngraph::opset1::ReduceMax>(node);
            });
        }

        auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
            if (std::dynamic_pointer_cast<const ngraph::opset6::RNNCell>(node)) {
                return false;
            } else if (std::dynamic_pointer_cast<const ngraph::opset6::GRUCell>(node)) {
                return false;
            } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ngraph::opset6::LSTMCell>(node)) {
                return lstm_cell->get_clip() == 0.0f && lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
            } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ngraph::opset1::LSTMCell>(node)) {
                return lstm_cell_v1->get_clip() == 0.0f && lstm_cell_v1->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
            }
            return false;
        };

        // Sequences supported by the plugin shouldn't be converted to TensorIterator.
        // sequence_length input is not supported in all Sequences, so if is_seq_len_provided() == true, we
        // should always convert to TensorIterator.
        // RNN/GRU Sequences are not supported in GPU plugin
        // LSTM Sequence supported with clip == 0, and activations have default values (sigmoid, tanh, tanh)
        auto isSequencePrimitiveSupported = [](const_node_ptr &node) -> bool {
            const auto& data = node->input(0);
            const auto& data_pshape = data.get_partial_shape();
            if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > 1 && !data_pshape[1].is_static())
                return false;
            auto max_seq_len = data.get_shape().at(1);
            if (std::dynamic_pointer_cast<const ngraph::opset6::RNNSequence>(node)) {
                return false;
            } else if (std::dynamic_pointer_cast<const ngraph::opset6::GRUSequence>(node)) {
                return false;
            } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ngraph::opset6::LSTMSequence>(node)) {
                return lstm_seq->get_clip() == 0.0f &&
                       lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                       max_seq_len < 16 &&
                       !ov::op::util::is_seq_len_provided(lstm_seq->get_input_node_shared_ptr(3),
                                                              max_seq_len);
            }
            return false;
        };

        pass_config->set_callback<ov::pass::RNNCellDecomposition,
                                  ov::pass::GRUCellDecomposition,
                                  ov::pass::LSTMCellDecomposition>(
            [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                return isCellPrimitiveSupported(node);
            });

        if (enable_loop_unrolling) {
            pass_config->set_callback<ov::pass::ConvertRNNSequenceToTensorIterator,
                    ov::pass::ConvertGRUSequenceToTensorIterator,
                    ov::pass::ConvertLSTMSequenceToTensorIterator>(
                    [isSequencePrimitiveSupported](const_node_ptr &node) -> bool {
                        return isSequencePrimitiveSupported(node);
                    });
        }


        pass_config->set_callback<ov::pass::MVN6Decomposition>(
            [](const_node_ptr &node) -> bool {
                const auto mvn = std::dynamic_pointer_cast<const ngraph::op::v6::MVN>(node);
                if (mvn != nullptr && node->get_input_size() == 2) {
                    if (auto axes_node = dynamic_cast<ngraph::op::v0::Constant*>(mvn->get_input_node_ptr(1))) {
                        auto mvn_axes = axes_node->cast_vector<int64_t>();
                        auto out_rank = mvn->get_output_partial_shape(0).size();

                        ov::normalize_axes(mvn.get(), out_rank, mvn_axes);
                        std::sort(mvn_axes.begin(), mvn_axes.end());

                        // Supported cases:
                        // 2 <= out_rank <= 5
                        // axes set: [out_rank - 1, out_rank - 2, ... r] where r > 1
                        // basically impl supports cases when tensor can be reshaped to [d1, d2]
                        // so that d2 is set of dimensions for normalization

                        // Skip unsupported ranks
                        if (out_rank == 1 || out_rank > 5)
                            return false;

                        // check axes set
                        for (size_t i = 0; i < mvn_axes.size(); i++) {
                            auto axis = mvn_axes[mvn_axes.size() - i - 1];
                            if (axis != static_cast<int64_t>(out_rank - i - 1) || axis == 0) {
                                  return false;
                            }
                        }
                        return true;
                    }
                }
                return false;
            });

        pass_config->enable<ov::pass::NormalizeL2Decomposition>();
        pass_config->set_callback<ov::pass::NormalizeL2Decomposition>(
            [](const_node_ptr &node) -> bool {
            // Condition to filter out axes such as [0, 1, 2] which is not supported currently.
            const auto norm = ov::as_type_ptr<const ngraph::op::v0::NormalizeL2>(node);
            const auto inputRank = norm->get_input_partial_shape(0).size();
            auto axesNode = ov::as_type_ptr<const ngraph::op::v0::Constant>(norm->get_input_node_shared_ptr(1));
            const auto axes = axesNode->cast_vector<size_t>();
            const auto isSupportedAxes = [](const std::vector<size_t> &axes, const size_t inputRank) {
                if (axes.size() == 1 && axes[0] == 1) {
                    return true;
                } else if (axes.size() == inputRank - 1) {
                    auto sortAxes = axes;
                    std::sort(sortAxes.begin(), sortAxes.end());
                    for (size_t i = 0; i < sortAxes.size(); i++) {
                        if (sortAxes[i] != i + 1)
                            return false;
                    }
                    return true;
                }
                return false;
            };

            if (!isSupportedAxes(axes, inputRank) && ngraph::shape_size(axesNode->get_shape()) != 0) {
                return false;
            }
            return true;
            });

        pass_config->enable<ov::pass::SoftmaxDecomposition>();
        pass_config->set_callback<ov::pass::SoftmaxDecomposition>(
            [](const_node_ptr &node) -> bool {
                return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
            });

        // List of enabled/disabled transformations
        pass_config->disable<ov::pass::ConvertGELU>();
        pass_config->disable<ov::pass::Gelu7Downgrade>();
        pass_config->disable<ov::pass::ConvertMod>();
        pass_config->disable<ov::pass::ConvertShuffleChannels3>();
        pass_config->disable<ov::pass::HSwishDecomposition>();
        pass_config->disable<ov::pass::HSigmoidDecomposition>();
        pass_config->disable<ov::pass::ReduceL1Decomposition>();
        pass_config->disable<ov::pass::ReduceL2Decomposition>();
        pass_config->disable<ov::pass::SoftPlusDecomposition>();
        pass_config->disable<ov::pass::LogSoftmaxDecomposition>();
        pass_config->disable<ov::pass::ConvertBroadcast3>();
        pass_config->disable<ov::pass::WeightsDequantizeToFakeQuantize>();
        pass_config->disable<ov::pass::SimplifyCTCGreedyDecoderSeqLen>();
        pass_config->disable<ov::pass::ConvertSoftMax8ToSoftMax1>();
        pass_config->disable<ov::pass::ConvertShapeOf3>();
        pass_config->disable<ov::pass::ConvertGather8ToGather7>();
        pass_config->disable<ov::pass::ConvertGather7ToGather1>();
        pass_config->disable<ov::pass::ConvertTopK11ToTopK3>();

        pass_config->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();

        if (enableInt8) {
            pass_config->set_callback<ov::pass::ConvertQuantizeDequantize>([&defaultPrecisions](const_node_ptr &node) -> bool {
                return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
            });
        }

        manager.run_passes(func);
    }

    if (enableInt8) {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply::lpt");
        using namespace ngraph::pass::low_precision;

        auto supportedPrecisions = std::vector<PrecisionsRestriction>({
            PrecisionsRestriction::create<ngraph::opset1::Convolution>({
                {{0}, {ngraph::element::u8, ngraph::element::i8}},
                {{1}, {ngraph::element::i8}},
            }),
            PrecisionsRestriction::create<ngraph::opset1::ConvolutionBackpropData>({
                {{0}, {ngraph::element::u8, ngraph::element::i8}},
                {{1}, {ngraph::element::i8}}
            }),
            PrecisionsRestriction::create<ngraph::opset1::GroupConvolution>({
                {{0}, {ngraph::element::u8, ngraph::element::i8}},
                {{1}, {ngraph::element::i8}}
            }),
            PrecisionsRestriction::create<ngraph::opset5::LSTMSequence>(PrecisionsRestriction::PrecisionsByPorts{}),
            PrecisionsRestriction::create<ngraph::opset6::GRUSequence>(PrecisionsRestriction::PrecisionsByPorts{})
        });

        auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
            QuantizationGranularityRestriction::create<ngraph::opset1::Convolution>({0}),
            QuantizationGranularityRestriction::create<ngraph::opset1::ConvolutionBackpropData>({0}),
        });

        ngraph::pass::Manager lptManager;

        auto lptPassConfig = lptManager.get_pass_config();
        // quantized LSTMSequence / GPUSequence are not supported yet. Avoid extra transformation
        lptPassConfig->disable<ngraph::pass::low_precision::RecurrentCellTransformation>();
        lptPassConfig->set_callback<ngraph::pass::low_precision::MarkupPrecisions>([](const_node_ptr& node) -> bool {
            if (const auto mulitply = std::dynamic_pointer_cast<const ngraph::opset1::Multiply>(node)) {
                return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
            }
            return false;
        });
        lptPassConfig->set_callback<ConvolutionBackpropDataTransformation>([func, defaultPrecisions](const_node_ptr& node) -> bool {
            auto fillStaticChannel = [func](const ngraph::PartialShape& shape, size_t& channel) -> bool {
                const auto rank = shape.rank();
                if (rank.is_dynamic()) {
                    return false;
                }
                if (rank.get_length() < 2l) {
                    return false;
                }
                const auto dimension = shape[1];
                if (dimension.is_dynamic()) {
                    return false;
                }
                channel = dimension.get_length();
                return true;
            };

            size_t inputChannels = 0;
            if (!fillStaticChannel(node->get_input_partial_shape(0), inputChannels)) {
                return true;
            }

            size_t outputChannels = 0;
            if (!fillStaticChannel(node->get_output_partial_shape(0), outputChannels)) {
                return true;
            }


            if ((inputChannels % 4 != 0) || (outputChannels % 16 != 0)) {
                return true;
            }

            return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions)
                || WeightableLayerTransformation::isAsymmetricOnWeights(node, defaultPrecisions);
        });

        lptPassConfig->set_callback<MultiplyToGroupConvolutionTransformation>([&](const_node_ptr& node) -> bool {
            // disable MultiplyToGroupConvolution if Multiply with Constant can be fused

            const auto dequantization = NetworkHelper::getDequantization(node, defaultPrecisions, 0, true);
            std::shared_ptr<ov::Node> parent = dequantization.empty() ? nullptr : dequantization.data.get_node()->shared_from_this();
            if (parent == nullptr) {
                const auto constantNode = NetworkHelper::getConstantInput(node);
                const auto constant = constantNode == nullptr ? nullptr : ngraph::as_type_ptr<ngraph::opset1::Constant>(constantNode);
                if (constant != nullptr) {
                    auto parent = node->get_input_node_shared_ptr(0);
                    if (parent == constant) {
                        parent = node->get_input_node_shared_ptr(1);
                    }
                }
            }

            if (parent != nullptr) {
                const auto parentHasOneConsumer = parent->get_output_target_inputs(0).size() == 1ul;
                if (parentHasOneConsumer) {
                    return true;
                }
            }

            // disable MultiplyToGroupConvolution for Multiply with scalar

            if (MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node)) {
                return true;
            }

            return false;
        });

        bool reshapeIgnorePerTensorQuantizationCheck = false;
        if (device_info.supports_immad) // Disable reshape transform until onednn i8 fc is optimized
            reshapeIgnorePerTensorQuantizationCheck = true;
        auto params = LayerTransformation::Params(true, element::f32, defaultPrecisions, reshapeIgnorePerTensorQuantizationCheck);
        lptManager.register_pass<LowPrecision>(supportedPrecisions, perTensorQuantization, params);
        lptManager.run_passes(func);
    }

    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::apply::run_passes");
        ngraph::pass::Manager manager;
        // This ConstantFolding pass is added to fold reshapes added for constant inputs on NMS internal operation which prevents upper-bound calculation
        // TODO: check why we have these reshapes
        manager.register_pass<ngraph::pass::ConstantFolding>();

        manager.register_pass<ov::pass::UnrollTensorIterator>();
        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<ov::pass::UnrollTensorIterator>(
            [enable_loop_unrolling](const std::shared_ptr<const ngraph::Node> &node) -> bool {
                auto sub_graph_op = std::dynamic_pointer_cast<const ngraph::op::util::SubGraphOp>(node);
                int64_t num_iter = sub_graph_op->get_num_iterations();
                if (!enable_loop_unrolling)
                    return num_iter != 1;
                return num_iter >= 16;
            });
        manager.register_pass<ov::pass::ResolveNameCollisions>();

        manager.run_passes(func);
    }
}
}  // namespace intel_gpu
}  // namespace ov
