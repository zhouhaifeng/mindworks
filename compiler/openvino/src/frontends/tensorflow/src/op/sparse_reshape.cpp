// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/sparse_fill_empty_rows.hpp"
#include "helper_ops/sparse_segment_ops.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_sparse_reshape_op(const ov::frontend::tensorflow::NodeContext& node) {
    // Currently, the translation for SparseReshape is possible only if new shape value is the same as the input shape
    // value or it is different just by one dynamic dimension of the new shape that can be replace with the
    // corresponding static dimension of the input shape.
    default_op_checks(node, 3, {"SparseReshape"});
    auto input_indices = node.get_input(0);
    auto input_shape = node.get_input(1);
    auto new_shape = node.get_input(2);

    // TODO: need to provide the full constant folding for WD both new_shape inputs of SparseReshape
    /*
    auto input_shape_const = get_constant_from_source(input_shape);
    auto new_shape_const = get_constant_from_source(new_shape);
    TENSORFLOW_OP_VALIDATION(
        node,
        input_shape_const && new_shape_const,
        "This case with SparseReshape is not possible to translate to OpenVINO opset. The input "
        "and new shapes of the input sparse tensor must be folded to constant values for SparseReshape translation.");
    auto input_shape_value = input_shape_const->cast_vector<int64_t>();
    auto new_shape_value = new_shape_const->cast_vector<int64_t>();

    TENSORFLOW_OP_VALIDATION(
        node,
        input_shape_value.size() == new_shape_value.size(),
        "This case with SparseReshape is not possible to translate to OpenVINO opset. The input "
        "and new shapes of the input sparse tensor must be of the same size for SparseReshape translation.");

    auto input_rank = input_shape_value.size();
    int num_dynamic_dimensions = 0;
    for (int64_t dim_ind = 0; dim_ind < input_rank; ++dim_ind) {
        TENSORFLOW_OP_VALIDATION(
            node,
            input_shape_value[dim_ind] >= 0,
            "This case with SparseReshape is not possible to translate to OpenVINO opset. The input shape of the "
            "sparse tensor must be fully static for SparseReshape translation.");
        if (new_shape_value[dim_ind] >= 0) {
            TENSORFLOW_OP_VALIDATION(
                node,
                input_shape_value[dim_ind] == new_shape_value[dim_ind],
                "This case with SparseReshape is not possible to translate to OpenVINO opset. Static dimensions of the "
                "input and new shapes must be equal for SparseReshape translation.");
        } else {
            ++num_dynamic_dimensions;
        }
    }

    TENSORFLOW_OP_VALIDATION(node,
                             num_dynamic_dimensions < 2,
                             "This case with SparseReshape is not possible to translate to OpenVINO opset. The number "
                             "of dynamic shapes in new shape must be 1 at most.");
    */
    return {input_indices, input_shape};
}

OutputVector translate_sparse_fill_empty_rows_op(const ov::frontend::tensorflow::NodeContext& node) {
    default_op_checks(node, 3, {"SparseFillEmptyRows"});
    auto input_indices = node.get_input(0);
    auto input_values = node.get_input(1);
    auto dense_shape = node.get_input(2);
    auto default_value = node.get_input(3);

    auto sparse_fill_empty_rows = make_shared<ov::frontend::tensorflow::SparseFillEmptyRows>(input_indices,
                                                                                             input_values,
                                                                                             dense_shape,
                                                                                             default_value,
                                                                                             node.get_decoder());
    set_node_name(node.get_name(), sparse_fill_empty_rows);
    return sparse_fill_empty_rows->outputs();
}

OutputVector translate_sparse_segment_sum_op(const ov::frontend::tensorflow::NodeContext& node) {
    auto input_size = node.get_input_size();
    TENSORFLOW_OP_VALIDATION(node,
                             input_size == 3 || input_size == 4,
                             "SparseSegmentSum must have either 3 or 4 inputs.");
    auto data = node.get_input(0);
    auto indices = node.get_input(1);
    auto segment_ids = node.get_input(2);

    std::shared_ptr<ov::frontend::tensorflow::SparseSegmentSum> sparse_segment_sum = nullptr;
    if (input_size == 3) {
        sparse_segment_sum =
            make_shared<ov::frontend::tensorflow::SparseSegmentSum>(data, indices, segment_ids, node.get_decoder());

    } else {
        auto num_segments = node.get_input(3);
        sparse_segment_sum = make_shared<ov::frontend::tensorflow::SparseSegmentSum>(data,
                                                                                     indices,
                                                                                     segment_ids,
                                                                                     num_segments,
                                                                                     node.get_decoder());
    }

    set_node_name(node.get_name(), sparse_segment_sum);
    return sparse_segment_sum->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
