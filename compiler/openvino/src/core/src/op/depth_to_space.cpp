// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/depth_to_space.hpp"

#include <cmath>
#include <cstddef>
#include <depth_to_space_shape_inference.hpp>
#include <memory>
#include <ngraph/op/constant.hpp>
#include <ngraph/ops.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/depth_to_space.hpp"
#include "ngraph/shape.hpp"
#include "openvino/core/validation_util.hpp"

using namespace ngraph;

op::DepthToSpace::DepthToSpace(const Output<Node>& data, const DepthToSpaceMode& mode, const size_t block_size)
    : Op({data}),
      m_blocksize(block_size),
      m_mode(mode) {
    constructor_validate_and_infer_types();
}

op::DepthToSpace::DepthToSpace(const Output<Node>& data, const std::string& mode, const size_t block_size)
    : DepthToSpace(data, as_enum<DepthToSpaceMode>(mode), block_size) {}

bool op::DepthToSpace::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_DepthToSpace_visit_attributes);
    visitor.on_attribute("block_size", m_blocksize);
    visitor.on_attribute("mode", m_mode);
    return true;
}

std::shared_ptr<Node> op::DepthToSpace::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_DepthToSpace_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<DepthToSpace>(new_args.at(0), m_mode, m_blocksize);
}

void op::DepthToSpace::validate_and_infer_types() {
    OV_OP_SCOPE(v0_DepthToSpace_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shape = shape_infer(this, get_node_input_partial_shapes(*this)).front();
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, get_input_element_type(0), output_shape);
}

namespace {
bool evaluate_depth_to_space(const HostTensorVector& outputs,
                             const HostTensorVector& inputs,
                             const std::size_t block_size,
                             const op::DepthToSpace::DepthToSpaceMode mode) {
    const auto& in = inputs[0];
    const auto& out = outputs[0];
    const size_t elem_size = in->get_element_type().size();
    if (in->get_partial_shape().is_dynamic()) {
        return false;
    }
    runtime::reference::depth_to_space(in->get_data_ptr<char>(),
                                       in->get_shape(),
                                       out->get_data_ptr<char>(),
                                       out->get_shape(),
                                       block_size,
                                       mode,
                                       elem_size);
    return true;
}
}  // namespace

bool op::DepthToSpace::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_DepthToSpace_evaluate);
    return evaluate_depth_to_space(outputs, inputs, m_blocksize, m_mode);
}

bool op::DepthToSpace::has_evaluate() const {
    OV_OP_SCOPE(v0_DepthToSpace_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic();
}

std::ostream& ov::operator<<(std::ostream& s, const ov::op::v0::DepthToSpace::DepthToSpaceMode& type) {
    return s << as_string(type);
}

void op::v0::DepthToSpace::set_block_size(size_t block_size) {
    m_blocksize = block_size;
}

void op::v0::DepthToSpace::set_mode(DepthToSpaceMode mode) {
    m_mode = mode;
}

namespace ov {
template <>
NGRAPH_API EnumNames<ngraph::op::DepthToSpace::DepthToSpaceMode>&
EnumNames<ngraph::op::DepthToSpace::DepthToSpaceMode>::get() {
    static auto enum_names = EnumNames<ngraph::op::DepthToSpace::DepthToSpaceMode>(
        "op::DepthToSpace::DepthToSpaceMode",
        {{"blocks_first", ngraph::op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST},
         {"depth_first", ngraph::op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST}});
    return enum_names;
}
}  // namespace ov
