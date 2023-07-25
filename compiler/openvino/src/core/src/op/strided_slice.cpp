// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/strided_slice.hpp"

#include <algorithm>

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/strided_slice.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "strided_slice_shape_inference.hpp"

using namespace std;
using namespace ngraph;

op::v1::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const Output<Node>& strides,
                                   const std::vector<int64_t>& begin_mask,
                                   const std::vector<int64_t>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : Op({data, begin, end, strides}),
      m_begin_mask{begin_mask},
      m_end_mask{end_mask},
      m_new_axis_mask{new_axis_mask},
      m_shrink_axis_mask{shrink_axis_mask},
      m_ellipsis_mask{ellipsis_mask} {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    ov::mark_as_precision_sensitive(input(3));
    constructor_validate_and_infer_types();
}

namespace {
shared_ptr<Node> calculate_default_strides(const Output<Node>& begin, const Output<Node>& end) {
    const auto begin_pshape = begin.get_partial_shape();
    const auto end_pshape = end.get_partial_shape();

    size_t strides_length = 0;
    if (begin_pshape.rank().is_static() && begin_pshape.rank().get_length() == 1 && begin_pshape[0].is_static()) {
        strides_length = begin_pshape[0].get_length();
    } else if (end_pshape.rank().is_static() && end_pshape.rank().get_length() == 1 && end_pshape[0].is_static()) {
        strides_length = end_pshape[0].get_length();
    } else  // dynamic case
    {
        NGRAPH_CHECK(begin_pshape.rank().is_static() && begin_pshape.rank().get_length() == 1,
                     "Begin input must be 1D");
        return std::make_shared<op::v1::Broadcast>(op::Constant::create(element::i64, {}, {1}),
                                                   std::make_shared<op::ShapeOf>(begin));
    }

    return op::Constant::create(element::i64, ov::Shape{strides_length}, vector<int64_t>(strides_length, 1));
}

/**
 * @brief Check if all indices in 1-D input shape are ignored by masks.
 *
 * @param shape        Indices shape (assume compatible 1-D shape).
 * @param ignored_mask Axis set of ignored indices.
 * @return True if all ignored other wise false.
 */
bool all_indices_ignored(const ov::PartialShape& shape, const std::vector<int64_t>& ignore_mask) {
    auto ignored = shape.rank().is_static() && ov::cmp::le(shape[0].get_interval().get_max_val(), ignore_mask.size());
    for (size_t i = 0; ignored && i < static_cast<size_t>(shape[0].get_interval().get_max_val()); ++i) {
        ignored = static_cast<bool>(ignore_mask[i]);
    }
    return ignored;
}
}  // namespace

op::v1::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const std::vector<int64_t>& begin_mask,
                                   const std::vector<int64_t>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : StridedSlice(data,
                   begin,
                   end,
                   calculate_default_strides(begin, end),
                   begin_mask,
                   end_mask,
                   new_axis_mask,
                   shrink_axis_mask,
                   ellipsis_mask) {}

bool op::v1::StridedSlice::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_StridedSlice_visit_attributes);
    visitor.on_attribute("begin_mask", m_begin_mask);
    visitor.on_attribute("end_mask", m_end_mask);
    visitor.on_attribute("new_axis_mask", m_new_axis_mask);
    visitor.on_attribute("shrink_axis_mask", m_shrink_axis_mask);
    visitor.on_attribute("ellipsis_mask", m_ellipsis_mask);
    return true;
}

void op::v1::StridedSlice::validate_and_infer_types() {
    OV_OP_SCOPE(v1_StridedSlice_validate_and_infer_types);
    const auto& begin_mask_et = get_input_element_type(1);
    const auto& end_mask_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          begin_mask_et.is_integral_number(),
                          "Begin mask must be an integral number, but is: ",
                          begin_mask_et);
    NODE_VALIDATION_CHECK(this,
                          end_mask_et.is_integral_number(),
                          "End mask must be an integral number, but is: ",
                          end_mask_et);

    constexpr auto are_mask_elem_in_range = cmp::Between<int64_t, cmp::BOTH>(0, 1);
    NODE_VALIDATION_CHECK(
        this,
        std::all_of(m_begin_mask.begin(), m_begin_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_end_mask.begin(), m_end_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_new_axis_mask.begin(), m_new_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_shrink_axis_mask.begin(), m_shrink_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_ellipsis_mask.begin(), m_ellipsis_mask.end(), are_mask_elem_in_range),
        "All masks of StridedSlice must have be 0 or 1");

    const vector<size_t> attr_sizes = {m_begin_mask.size(),
                                       m_end_mask.size(),
                                       m_new_axis_mask.size(),
                                       m_shrink_axis_mask.size(),
                                       m_ellipsis_mask.size()};
    const auto are_attr_sizes_eq = std::all_of(attr_sizes.begin(), attr_sizes.end(), [&attr_sizes](size_t s) {
        return (s == 0) || (attr_sizes[0] == s);
    });
    NODE_VALIDATION_CHECK(this, are_attr_sizes_eq, "All masks of StridedSlice must have the same size");

    // Fill up strides input with default strides if not set by this point.
    if (get_input_size() < 4) {
        set_argument(3, calculate_default_strides(get_input_node_ptr(1)->output(0), get_input_node_ptr(2)->output(0)));
    }

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

AxisSet op::v1::StridedSlice::convert_mask_to_axis_set(const std::vector<int64_t>& mask) const {
    AxisSet axis_set{};
    for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
        if (mask[i] == 1) {
            axis_set.emplace(i);
        }
    }
    return axis_set;
}

shared_ptr<Node> op::v1::StridedSlice::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_StridedSlice_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::StridedSlice>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         m_begin_mask,
                                         m_end_mask,
                                         m_new_axis_mask,
                                         m_shrink_axis_mask,
                                         m_ellipsis_mask);
}

namespace strided_slice {
namespace {
OPENVINO_SUPPRESS_DEPRECATED_START
inline bool evaluate(const HostTensorPtr& in, const SlicePlan& sp, const HostTensorPtr& out)

{
    auto in_shape = in->get_shape();
    out->set_shape(sp.reshape_out_shape);
    runtime::reference::strided_slice(in->get_data_ptr<char>(),
                                      out->get_data_ptr<char>(),
                                      in_shape,
                                      sp,
                                      in->get_element_type().size());
    return true;
}

bool evaluate_strided_slice(const HostTensorPtr& in,
                            const HostTensorPtr& begin,
                            const HostTensorPtr& end,
                            const HostTensorPtr& stride,
                            const AxisSet& begin_mask,
                            const AxisSet& end_mask,
                            const AxisSet& new_axis_mask,
                            const AxisSet& shrink_axis_mask,
                            const AxisSet& ellipsis_mask,
                            const HostTensorPtr& out) {
    std::vector<int64_t> begin_const = host_tensor_2_vector<int64_t>(begin);
    std::vector<int64_t> end_const = host_tensor_2_vector<int64_t>(end);
    std::vector<int64_t> stride_const = host_tensor_2_vector<int64_t>(stride);
    SlicePlan slice_plan = make_slice_plan(in->get_shape(),
                                           begin_const,
                                           end_const,
                                           stride_const,
                                           begin_mask,
                                           end_mask,
                                           new_axis_mask,
                                           shrink_axis_mask,
                                           ellipsis_mask);
    return evaluate(in, slice_plan, out);
}
OPENVINO_SUPPRESS_DEPRECATED_END
}  // namespace
}  // namespace strided_slice

bool op::v1::StridedSlice::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v1_StridedSlice_evaluate);
    // FIXME: 4th input is optional, but it is required by the following code
    OPENVINO_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 4));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return strided_slice::evaluate_strided_slice(input_values[0],
                                                 input_values[1],
                                                 input_values[2],
                                                 input_values[3],
                                                 convert_mask_to_axis_set(get_begin_mask()),
                                                 convert_mask_to_axis_set(get_end_mask()),
                                                 convert_mask_to_axis_set(get_new_axis_mask()),
                                                 convert_mask_to_axis_set(get_shrink_axis_mask()),
                                                 convert_mask_to_axis_set(get_ellipsis_mask()),
                                                 output_values[0]);
}

bool op::v1::StridedSlice::has_evaluate() const {
    OV_OP_SCOPE(v1_StridedSlice_has_evaluate);
    return get_input_size() == 4;
}

bool op::v1::StridedSlice::indices_input_has_and_set_bounds(const size_t port, const std::vector<int64_t>& mask) const {
    const auto& lb_t = get_input_tensor(port).get_lower_value();
    const auto& ub_t = get_input_tensor(port).get_upper_value();

    const auto mask_set = convert_mask_to_axis_set(mask);
    bool valid_bounds = all_indices_ignored(get_input_partial_shape(port), mask);

    if (!valid_bounds && lb_t && ub_t) {
        using TCast = int64_t;
        constexpr auto i64_cast = ov::util::Cast<TCast>();
        const auto lb = ov::get_tensor_data_as<TCast>(lb_t, i64_cast);
        const auto ub = ov::get_tensor_data_as<TCast>(ub_t, i64_cast);

        size_t axis = 0;
        valid_bounds =
            std::equal(lb.cbegin(), lb.cend(), ub.cbegin(), [&axis, &mask_set](TCast lhs, TCast rhs) -> bool {
                return mask_set.count(axis++) || lhs == rhs;
            });
    }

    return valid_bounds;
}

bool op::v1::StridedSlice::evaluate_lower(ov::TensorVector& output_values) const {
    return indices_input_has_and_set_bounds(1, get_begin_mask()) &&
           indices_input_has_and_set_bounds(2, get_end_mask()) && get_input_tensor(3).has_and_set_bound() &&
           default_lower_bound_evaluator(this, output_values);
}

bool op::v1::StridedSlice::evaluate_upper(ov::TensorVector& output_values) const {
    return indices_input_has_and_set_bounds(1, get_begin_mask()) &&
           indices_input_has_and_set_bounds(2, get_end_mask()) && get_input_tensor(3).has_and_set_bound() &&
           default_upper_bound_evaluator(this, output_values);
}

bool op::v1::StridedSlice::evaluate_label(TensorLabelVector& output_labels) const {
    return indices_input_has_and_set_bounds(1, get_begin_mask()) &&
           indices_input_has_and_set_bounds(2, get_end_mask()) && get_input_tensor(3).has_and_set_bound() &&
           default_label_evaluator(this, {0}, output_labels);
}

bool op::v1::StridedSlice::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    auto is_folded = Node::constant_fold(output_values, inputs_values);
    if (!is_const_fold_disabled() && !is_folded) {
        // If all ignored mask are set for all begin or end then replace this input by dummy constant
        // to avoid return false from `could_propagate` during bound evaluation (value of const will be ignored).
        auto get_indices_input = [&inputs_values](size_t port, const std::vector<int64_t>& mask) -> Output<Node> {
            const auto& port_shape = inputs_values[port].get_partial_shape();
            const auto& data_shape = inputs_values[0].get_partial_shape();

            size_t size;
            if (port_shape.rank().is_static() && port_shape[0].is_static())
                size = static_cast<size_t>(port_shape[0].get_length());
            else if (data_shape.rank().is_static())
                size = data_shape.size();
            else
                size = mask.size();

            const auto& zero_constant =
                make_shared<ov::opset1::Constant>(inputs_values[port].get_element_type(), ov::Shape{size}, 0);
            return all_indices_ignored(inputs_values[port].get_partial_shape(), mask) ? zero_constant
                                                                                      : inputs_values[port];
        };

        const auto& begin = get_indices_input(1, get_begin_mask());
        const auto& end = get_indices_input(2, get_end_mask());

        const auto& output =
            ((&begin != &inputs_values[1]) || (&end != &inputs_values[2]))
                ? clone_with_new_inputs(OutputVector{inputs_values[0], begin, end, inputs_values[3]})->output(0)
                : this->output(0);

        std::vector<Node*> nodes;
        // Check if bounds can be evaluated and none of output nodes have disabled constant folding.
        if (ov::could_propagate(output, nodes) && std::none_of(nodes.begin(), nodes.end(), [](const Node* n) {
                return ov::pass::constant_folding_is_disabled(n);
            })) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            if (const auto c = ov::get_constant_from_source(output)) {
                OPENVINO_SUPPRESS_DEPRECATED_END
                output_values[0] = c;
                auto output_ptr = output_values[0].get_node_shared_ptr();
                for (const auto& n : nodes) {
                    copy_runtime_info(n->shared_from_this(), output_ptr);
                }
                is_folded = true;
            }
        }
    }
    return is_folded;
}
