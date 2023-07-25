// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/primitives/matrix_nms.hpp"

#include <memory>
#include <openvino/opsets/opset8.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"

namespace ov {
namespace op {
namespace internal {
using NmsStaticShapeIE8 = ov::op::internal::NmsStaticShapeIE<ov::opset8::MatrixNms>;
}
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

namespace {
void CreateNmsStaticShapeIE8Op(Program& p, const std::shared_ptr<ov::op::internal::NmsStaticShapeIE8>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);

    std::vector<cldnn::memory::ptr> shared_memory;

    auto outputIndices = op->get_output_shape(0)[0];
    cldnn::layout mutableLayoutFirst = cldnn::layout(cldnn::element_type_to_data_type(ngraph::element::i32),
                                                     cldnn::format::bfyx,
                                                     cldnn::tensor(static_cast<int32_t>(outputIndices), 1, 1, 1));

    shared_memory.emplace_back(p.get_engine().allocate_memory(mutableLayoutFirst));

    cldnn::primitive_id matrix_nms_mutable_id_w_first = layer_type_name_ID(op) + "_md_write_first";
    auto matrix_nms_mutable_prim_first = cldnn::mutable_data(matrix_nms_mutable_id_w_first, shared_memory.back());
    p.add_primitive(*op, matrix_nms_mutable_prim_first);
    inputs.push_back(cldnn::input_info(matrix_nms_mutable_id_w_first));

    auto batches_num = op->get_output_shape(2)[0];
    cldnn::layout mutableLayoutSecond = cldnn::layout(cldnn::element_type_to_data_type(ngraph::element::i32),
                                                      cldnn::format::bfyx,
                                                      cldnn::tensor(static_cast<int32_t>(batches_num), 1, 1, 1));

    shared_memory.emplace_back(p.get_engine().allocate_memory(mutableLayoutSecond));

    cldnn::primitive_id matrix_nms_mutable_id_w_second = layer_type_name_ID(op) + "_md_write_second";
    auto matrix_nms_mutable_prim_second = cldnn::mutable_data(matrix_nms_mutable_id_w_second, shared_memory.back());
    p.add_primitive(*op, matrix_nms_mutable_prim_second);
    inputs.push_back(cldnn::input_info(matrix_nms_mutable_id_w_second));

    auto matrixNmsLayerName = layer_type_name_ID(op) + ".out0";

    auto prim = cldnn::matrix_nms(matrixNmsLayerName,
                                  inputs[0],
                                  inputs[1],
                                  inputs[inputs.size() - 2],
                                  inputs[inputs.size() - 1],
                                  op->get_attrs());

    p.add_primitive(*op, prim);

    cldnn::primitive_id matrix_nms_id_r_first = layer_type_name_ID(op) + ".out1";
    auto matrix_nms_mutable_prim_r_first =
        cldnn::mutable_data(matrix_nms_id_r_first, { cldnn::input_info(matrixNmsLayerName) }, shared_memory.front());
    p.add_primitive(*op, matrix_nms_mutable_prim_r_first);

    cldnn::primitive_id matrix_nms_id_r_second = layer_type_name_ID(op) + ".out2";
    auto matrix_nms_mutable_prim_r_second =
        cldnn::mutable_data(matrix_nms_id_r_second, { cldnn::input_info(matrixNmsLayerName) }, shared_memory.back());
    p.add_primitive(*op, matrix_nms_mutable_prim_r_second);
}

}  // anonymous namespace

REGISTER_FACTORY_IMPL(internal, NmsStaticShapeIE8);

}  // namespace intel_gpu
}  // namespace ov
