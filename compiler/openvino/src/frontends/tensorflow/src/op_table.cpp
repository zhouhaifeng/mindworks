// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "common_op_table.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

#define TF_OP_CONVERTER(op) OutputVector op(const ov::frontend::tensorflow::NodeContext& node)

TF_OP_CONVERTER(translate_assignvariable_op);
TF_OP_CONVERTER(translate_block_lstm_op);
TF_OP_CONVERTER(translate_fifo_queue_op);
TF_OP_CONVERTER(translate_gru_block_cell_op);
TF_OP_CONVERTER(translate_hash_table_op);
TF_OP_CONVERTER(translate_if_op);
TF_OP_CONVERTER(translate_iterator_get_next_op);
TF_OP_CONVERTER(translate_iterator_op);
TF_OP_CONVERTER(translate_merge_op);
TF_OP_CONVERTER(translate_mergev2checkpoint_op);
TF_OP_CONVERTER(translate_partitioned_call_op);
TF_OP_CONVERTER(translate_placeholder_linked_op);
TF_OP_CONVERTER(translate_queue_dequeue_op);
TF_OP_CONVERTER(translate_queue_dequeue_many_op);
TF_OP_CONVERTER(translate_readvariable_op);
TF_OP_CONVERTER(translate_restorev2_op);
TF_OP_CONVERTER(translate_sparse_fill_empty_rows_op);
TF_OP_CONVERTER(translate_sparse_reshape_op);
TF_OP_CONVERTER(translate_sparse_segment_sum_op);
TF_OP_CONVERTER(translate_staticregexfullmatch_op);
TF_OP_CONVERTER(translate_stringjoin_op);
TF_OP_CONVERTER(translate_switch_op);
TF_OP_CONVERTER(translate_varhandle_op);
TF_OP_CONVERTER(translate_variable_op);
TF_OP_CONVERTER(translate_varisinitialized_op);
TF_OP_CONVERTER(translate_while_op);

const std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        // note: UnaryOp translator declaration for each op must to be added in unary_op.cpp file
        {"Abs", CreatorFunction(translate_unary_op<opset8::Abs>)},
        {"Acos", CreatorFunction(translate_unary_op<opset8::Acos>)},
        {"Acosh", CreatorFunction(translate_unary_op<opset8::Acosh>)},
        {"Asin", CreatorFunction(translate_unary_op<opset8::Asin>)},
        {"Asinh", CreatorFunction(translate_unary_op<opset8::Asinh>)},
        {"Atan", CreatorFunction(translate_unary_op<opset8::Atan>)},
        {"Atanh", CreatorFunction(translate_unary_op<opset8::Atanh>)},
        {"Ceil", CreatorFunction(translate_unary_op<opset8::Ceiling>)},
        {"Cos", CreatorFunction(translate_unary_op<opset8::Cos>)},
        {"Cosh", CreatorFunction(translate_unary_op<opset8::Cosh>)},
        {"Erf", CreatorFunction(translate_unary_op<opset8::Erf>)},
        {"Exp", CreatorFunction(translate_unary_op<opset8::Exp>)},
        {"Floor", CreatorFunction(translate_unary_op<opset8::Floor>)},
        {"IsFinite", CreatorFunction(translate_unary_op<opset10::IsFinite>)},
        {"IsInf", CreatorFunction(translate_unary_op<opset10::IsInf>)},
        {"IsNan", CreatorFunction(translate_unary_op<opset10::IsNaN>)},
        {"Log", CreatorFunction(translate_unary_op<opset8::Log>)},
        {"LogicalNot", CreatorFunction(translate_unary_op<opset8::LogicalNot>)},
        {"Mish", CreatorFunction(translate_unary_op<opset8::Mish>)},
        {"Neg", CreatorFunction(translate_unary_op<opset8::Negative>)},
        {"Relu", CreatorFunction(translate_unary_op<opset8::Relu>)},
        {"Sigmoid", CreatorFunction(translate_unary_op<opset8::Sigmoid>)},
        {"Sin", CreatorFunction(translate_unary_op<opset8::Sin>)},
        {"Sinh", CreatorFunction(translate_unary_op<opset8::Sinh>)},
        {"Sign", CreatorFunction(translate_unary_op<opset8::Sign>)},
        {"Softplus", CreatorFunction(translate_unary_op<opset8::SoftPlus>)},
        {"Softsign", CreatorFunction(translate_unary_op<opset9::SoftSign>)},
        {"Tan", CreatorFunction(translate_unary_op<opset8::Tan>)},
        {"Tanh", CreatorFunction(translate_unary_op<opset8::Tanh>)},
        {"Swish", CreatorFunction(translate_unary_op<opset8::Swish>)},

        // note: BinaryOp translator declaration for each op must to be added in binary_op.cpp file
        {"Add", CreatorFunction(translate_binary_op<opset8::Add>)},
        {"AddV2", CreatorFunction(translate_binary_op<opset8::Add>)},
        {"Equal", CreatorFunction(translate_binary_op<opset8::Equal>)},
        {"FloorMod", CreatorFunction(translate_binary_op<opset8::FloorMod>)},
        {"Greater", CreatorFunction(translate_binary_op<opset8::Greater>)},
        {"GreaterEqual", CreatorFunction(translate_binary_op<opset8::GreaterEqual>)},
        {"Less", CreatorFunction(translate_binary_op<opset8::Less>)},
        {"LessEqual", CreatorFunction(translate_binary_op<opset8::LessEqual>)},
        {"LogicalAnd", CreatorFunction(translate_binary_op<opset8::LogicalAnd>)},
        {"LogicalOr", CreatorFunction(translate_binary_op<opset8::LogicalOr>)},
        {"LogicalXor", CreatorFunction(translate_binary_op<opset8::LogicalXor>)},
        {"Maximum", CreatorFunction(translate_binary_op<opset8::Maximum>)},
        {"Minimum", CreatorFunction(translate_binary_op<opset8::Minimum>)},
        {"Mul", CreatorFunction(translate_binary_op<opset8::Multiply>)},
        {"Mod", CreatorFunction(translate_binary_op<opset8::Mod>)},
        {"NotEqual", CreatorFunction(translate_binary_op<opset8::NotEqual>)},
        {"Pow", CreatorFunction(translate_binary_op<opset8::Power>)},
        {"RealDiv", CreatorFunction(translate_binary_op<opset8::Divide>)},
        {"SquaredDifference", CreatorFunction(translate_binary_op<opset8::SquaredDifference>)},
        {"Sub", CreatorFunction(translate_binary_op<opset8::Subtract>)},

        // note: ReduceOp translator declaration for each op must to be added in reduce.cpp file
        {"Any", CreatorFunction(translate_direct_reduce_op<opset8::ReduceLogicalOr>)},
        {"All", CreatorFunction(translate_direct_reduce_op<opset8::ReduceLogicalAnd>)},
        {"EuclideanNorm", CreatorFunction(translate_direct_reduce_op<opset8::ReduceL2>)},
        {"Max", CreatorFunction(translate_direct_reduce_op<opset8::ReduceMax>)},
        {"Mean", CreatorFunction(translate_direct_reduce_op<opset8::ReduceMean>)},
        {"Min", CreatorFunction(translate_direct_reduce_op<opset8::ReduceMin>)},
        {"Prod", CreatorFunction(translate_direct_reduce_op<opset8::ReduceProd>)},
        {"Sum", CreatorFunction(translate_direct_reduce_op<opset8::ReduceSum>)},

        // Separate translators:
        {"AddN", CreatorFunction(translate_add_n_op)},
        {"ArgMax", CreatorFunction(translate_arg_max_op)},
        {"ArgMin", CreatorFunction(translate_arg_min_op)},
        {"Assert", CreatorFunction(translate_no_op)},
        {"AvgPool", CreatorFunction(translate_avg_pool_op)},
        {"AvgPool3D", CreatorFunction(translate_avg_pool_op)},
        {"BatchMatMul", CreatorFunction(translate_batch_mat_mul_op)},
        {"BatchMatMulV2", CreatorFunction(translate_batch_mat_mul_op)},
        {"BatchToSpaceND", CreatorFunction(translate_batch_to_space_nd_op)},
        {"BroadcastArgs", CreatorFunction(translate_broadcast_args_op)},
        {"BroadcastTo", CreatorFunction(translate_broadcast_to_op)},
        {"Bucketize", CreatorFunction(translate_bucketize_op)},
        {"BiasAdd", CreatorFunction(translate_bias_add_op)},
        {"Cast", CreatorFunction(translate_cast_op)},
        {"ClipByValue", CreatorFunction(translate_clip_by_value_op)},
        {"Concat", CreatorFunction(translate_concat_op)},
        {"ConcatV2", CreatorFunction(translate_concat_op)},
        {"Const", CreatorFunction(translate_const_op)},
        {"Conv2D", CreatorFunction(translate_conv_2d_op)},
        {"Conv2DBackpropInput", CreatorFunction(translate_conv_2d_backprop_input_op)},
        {"Conv3D", CreatorFunction(translate_conv_3d_op)},
        {"Conv3DBackpropInputV2", CreatorFunction(translate_conv_3d_backprop_input_v2_op)},
        {"CropAndResize", CreatorFunction(translate_crop_and_resize_op)},
        {"CTCGreedyDecoder", CreatorFunction(translate_ctc_greedy_decoder_op)},
        {"CTCLoss", CreatorFunction(translate_ctc_loss_op)},
        {"Cumsum", CreatorFunction(translate_cumsum_op)},
        {"DepthToSpace", CreatorFunction(translate_depth_to_space_op)},
        {"DepthwiseConv2dNative", CreatorFunction(translate_depthwise_conv_2d_native_op)},
        {"DynamicPartition", CreatorFunction(translate_dynamic_partition_op)},
        {"Einsum", CreatorFunction(translate_einsum_op)},
        {"Elu", CreatorFunction(translate_elu_op)},
        {"EmptyTensorList", CreatorFunction(translate_tensor_list_reserve_op)},
        {"ExpandDims", CreatorFunction(translate_expand_dims_op)},
        {"ExtractImagePatches", CreatorFunction(translate_extract_image_patches_op)},
        {"FakeQuantWithMinMaxVars", CreatorFunction(translate_fake_quant_op)},
        {"FakeQuantWithMinMaxVarsPerChannel", CreatorFunction(translate_fake_quant_op)},
        {"FIFOQueue", CreatorFunction(translate_fifo_queue_op)},
        {"FIFOQueueV2", CreatorFunction(translate_fifo_queue_op)},
        {"Fill", CreatorFunction(translate_fill_op)},
        {"FloorDiv", CreatorFunction(translate_floor_div_op)},
        {"FusedBatchNorm", CreatorFunction(translate_fused_batch_norm_op)},
        {"FusedBatchNormV2", CreatorFunction(translate_fused_batch_norm_op)},
        {"FusedBatchNormV3", CreatorFunction(translate_fused_batch_norm_op)},
        {"Gather", CreatorFunction(translate_gather_op)},
        {"GatherV2", CreatorFunction(translate_gather_v2_op)},
        {"GatherNd", CreatorFunction(translate_gather_nd_op)},
        {"HashTable", CreatorFunction(translate_hash_table_op)},
        {"HashTableV2", CreatorFunction(translate_hash_table_op)},
        {"Identity", CreatorFunction(translate_identity_op)},
        {"IdentityN", CreatorFunction(translate_identity_n_op)},
        {"If", CreatorFunction(translate_if_op)},
        {"input_arg", CreatorFunction(translate_input_arg_op)},
        {"Iterator", CreatorFunction(translate_iterator_op)},
        {"IteratorGetNext", CreatorFunction(translate_iterator_get_next_op)},
        {"IteratorV2", CreatorFunction(translate_iterator_op)},
        {"output_arg", CreatorFunction(translate_output_arg_op)},
        {"L2Loss", CreatorFunction(translate_l2_loss_op)},
        {"LeakyRelu", CreatorFunction(translate_leaky_relu_op)},
        {"LinSpace", CreatorFunction(translate_linspace_op)},
        {"ListDiff", CreatorFunction(translate_list_diff_op)},
        {"LogSoftmax", CreatorFunction(translate_log_softmax_op)},
        {"Log1p", CreatorFunction(translate_log_1p_op)},
        {"LookupTableInsert", CreatorFunction(translate_no_op)},
        {"LookupTableInsertV2", CreatorFunction(translate_no_op)},
        {"LRN", CreatorFunction(translate_lrn_op)},
        {"MatMul", CreatorFunction(translate_mat_mul_op)},
        {"MatrixDiag", CreatorFunction(translate_matrix_diag_op)},
        {"MaxPool", CreatorFunction(translate_max_pool_op)},
        {"MaxPoolV2", CreatorFunction(translate_max_pool_op)},
        {"MaxPool3D", CreatorFunction(translate_max_pool_op)},
        {"Merge", CreatorFunction(translate_merge_op)},
        {"MirrorPad", CreatorFunction(translate_mirror_pad_op)},
        {"MutableHashTable", CreatorFunction(translate_hash_table_op)},
        {"MutableHashTableV2", CreatorFunction(translate_hash_table_op)},
        {"NonMaxSuppression", CreatorFunction(translate_non_max_suppression_op)},
        {"NonMaxSuppressionV2", CreatorFunction(translate_non_max_suppression_op)},
        {"NonMaxSuppressionV3", CreatorFunction(translate_non_max_suppression_op)},
        {"NonMaxSuppressionV4", CreatorFunction(translate_non_max_suppression_op)},
        {"NonMaxSuppressionV5", CreatorFunction(translate_non_max_suppression_op)},
        {"NoOp", CreatorFunction(translate_no_op)},  // do nothing
        {"OneHot", CreatorFunction(translate_one_hot_op)},
        {"OneShotIterator", CreatorFunction(translate_iterator_op)},
        {"Pack", CreatorFunction(translate_pack_op)},
        {"Pad", CreatorFunction(translate_pad_op)},
        {"PadV2", CreatorFunction(translate_padv2_op)},
        {"QueueDequeue", CreatorFunction(translate_queue_dequeue_op)},
        {"QueueDequeueV2", CreatorFunction(translate_queue_dequeue_op)},
        {"QueueDequeueUpTo", CreatorFunction(translate_queue_dequeue_many_op)},
        {"QueueDequeueUpToV2", CreatorFunction(translate_queue_dequeue_many_op)},
        {"QueueDequeueMany", CreatorFunction(translate_queue_dequeue_many_op)},
        {"DynamicStitch", CreatorFunction(translate_parallel_dynamic_stitch_op)},
        {"ParallelDynamicStitch", CreatorFunction(translate_parallel_dynamic_stitch_op)},
        {"PartitionedCall", CreatorFunction(translate_partitioned_call_op)},
        {"Placeholder", CreatorFunction(translate_placeholder_linked_op)},
        {"PlaceholderWithDefault", CreatorFunction(translate_placeholder_with_default_op)},
        {"PreventGradient", CreatorFunction(translate_identity_op)},
        {"Range", CreatorFunction(translate_range_op)},
        {"Rank", CreatorFunction(translate_rank_op)},
        {"RandomUniform", CreatorFunction(translate_random_uniform_op)},
        {"RandomUniformInt", CreatorFunction(translate_random_uniform_int_op)},
        {"Reciprocal", CreatorFunction(translate_reciprocal_op)},
        {"Relu6", CreatorFunction(translate_relu_6_op)},
        {"Reshape", CreatorFunction(translate_reshape_op)},
        {"Reverse", CreatorFunction(translate_reverse_op)},
        {"ReverseSequence", CreatorFunction(translate_reverse_sequence_op)},
        {"ReverseV2", CreatorFunction(translate_reverse_v2_op)},
        {"ResizeBilinear", CreatorFunction(translate_interpolate_op)},
        {"ResizeNearestNeighbor", CreatorFunction(translate_interpolate_op)},
        {"ResourceGather", CreatorFunction(translate_resource_gather_op)},
        {"Roll", CreatorFunction(translate_roll_op)},
        {"Round", CreatorFunction(translate_round_op)},
        {"Rsqrt", CreatorFunction(translate_rsqrt_op)},
        {"SaveV2", CreatorFunction(translate_no_op)},
        {"ScatterNd", CreatorFunction(translate_scatter_nd_op)},
        {"SegmentSum", CreatorFunction(translate_segment_sum_op)},
        {"SparseToDense", CreatorFunction(translate_sparse_to_dense_op)},
        {"Select", CreatorFunction(translate_select_op)},
        {"SelectV2", CreatorFunction(translate_select_v2_op)},
        {"Shape", CreatorFunction(translate_shape_op)},
        {"Size", CreatorFunction(translate_size_op)},
        {"Slice", CreatorFunction(translate_slice_op)},
        {"Snapshot", CreatorFunction(translate_identity_op)},
        {"Softmax", CreatorFunction(translate_softmax_op)},
        {"SpaceToDepth", CreatorFunction(translate_space_to_depth_op)},
        {"SparseReshape", CreatorFunction(translate_sparse_reshape_op)},
        {"Split", CreatorFunction(translate_split_op)},
        {"SplitV", CreatorFunction(translate_split_v_op)},
        {"StopGradient", CreatorFunction(translate_identity_op)},
        {"Sqrt", CreatorFunction(translate_sqrt_op)},
        {"Square", CreatorFunction(translate_square_op)},
        {"Squeeze", CreatorFunction(translate_squeeze_op)},
        {"SpaceToBatchND", CreatorFunction(translate_space_to_batch_nd_op)},
        {"StatefulPartitionedCall", CreatorFunction(translate_partitioned_call_op)},
        {"StatelessIf", CreatorFunction(translate_if_op)},
        {"StatelessWhile", CreatorFunction(translate_while_op)},
        {"StridedSlice", CreatorFunction(translate_strided_slice_op)},
        {"Switch", CreatorFunction(translate_switch_op)},
        {"TensorListFromTensor", CreatorFunction(translate_tensor_list_from_tensor_op)},
        {"TensorListGetItem", CreatorFunction(translate_tensor_list_get_item_op)},
        {"TensorListPushBack", CreatorFunction(translate_tensor_list_push_back_op)},
        {"TensorListSetItem", CreatorFunction(translate_tensor_list_set_item_op)},
        {"TensorListStack", CreatorFunction(translate_tensor_list_stack_op)},
        {"TensorListReserve", CreatorFunction(translate_tensor_list_reserve_op)},
        {"Tile", CreatorFunction(translate_tile_op)},
        {"TopK", CreatorFunction(translate_top_k_op)},
        {"TopKV2", CreatorFunction(translate_top_k_v2_op)},
        {"Transpose", CreatorFunction(translate_transpose_op)},
        {"Unpack", CreatorFunction(translate_unpack_op)},
        {"While", CreatorFunction(translate_while_op)},
        {"Where", CreatorFunction(translate_where_op)},
        {"Xdivy", CreatorFunction(translate_x_div_y_op)},
        {"ZerosLike", CreatorFunction(translate_zeros_like_op)},

        // Translators for SavedModel and MetaGraph
        {"Assign", CreatorFunction(translate_readvariable_op)},
        {"AssignVariableOp", CreatorFunction(translate_assignvariable_op)},
        {"IsVariableInitialized", CreatorFunction(translate_varisinitialized_op)},
        {"MergeV2Checkpoints", CreatorFunction(translate_identity_op)},
        {"ReadVariableOp", CreatorFunction(translate_readvariable_op)},
        {"RestoreV2", CreatorFunction(translate_restorev2_op)},
        {"ShardedFilename", CreatorFunction(translate_identity_op)},
        {"StaticRegexFullMatch", CreatorFunction(translate_staticregexfullmatch_op)},
        {"StringJoin", CreatorFunction(translate_stringjoin_op)},
        {"VarIsInitializedOp", CreatorFunction(translate_varisinitialized_op)},
        {"VarHandleOp", CreatorFunction(translate_varhandle_op)},
        {"VariableV2", CreatorFunction(translate_varhandle_op)},

        // Translator for Checkpoint V1
        {"Variable", CreatorFunction(translate_variable_op)},

        // Translators for internal operations
        {"BlockLSTM", CreatorFunction(translate_block_lstm_op)},
        {"GRUBlockCell", CreatorFunction(translate_gru_block_cell_op)},
        {"SparseFillEmptyRows", CreatorFunction(translate_sparse_fill_empty_rows_op)},
        {"SparseSegmentSum", CreatorFunction(translate_sparse_segment_sum_op)},
        {"Unique", CreatorFunction(translate_unique_op)},
    };
};
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov