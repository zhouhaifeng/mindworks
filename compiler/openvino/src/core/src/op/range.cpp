// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/range.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/range.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "range_shape_inference.hpp"

using namespace std;
using namespace ngraph;

//
// The code in the following three functions is a bit awkward, to work around some compiler
// warnings and the need to support our custom float16/bfloat16 type:
//
// (1) We can't use STL things like isnan, because our custom float16/bfloat16 types don't always
//     support them.
// (2) We check whether (x - x) == (x - x) to check for "is_finite".
// (3) We have to break (x - x) out into a temporary because otherwise the compiler throws a
//     warning about == on floats.
// (4) We check <0 || >0 to check for != 0, because otherwise the compiler throws a warning about
//     == on floats.
//
template <typename T>
static typename std::enable_if<std::is_integral<T>::value, bool>::type check_value(T value) {
    // Nothing to check for integral types.
    return true;
}

template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                   std::is_same<T, bfloat16>::value,
                               bool>::type
check_value(T value) {
    T value_minus_value = value - value;
    return value == value && value_minus_value == value_minus_value;
}

op::v4::Range::Range(const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step,
                     element::Type output_type)
    : Op({start, stop, step}),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v4::Range::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_Range_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v4::Range::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Range_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_output_type.is_integral_number() || m_output_type.is_real(),
                          "output tensor type should be a numeric type. Got: ",
                          m_output_type);

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_integral_number() || get_input_element_type(0).is_real(),
                          "'start' input scalar should be a numeric type. Got: ",
                          get_input_element_type(0));
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number() || get_input_element_type(1).is_real(),
                          "'stop' input scalar should be a numeric type. Got: ",
                          get_input_element_type(1));
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number() || get_input_element_type(2).is_real(),
                          "'step' input scalar should be a numeric type. Got: ",
                          get_input_element_type(2));

    std::vector<PartialShape> input_shapes;
    for (size_t i = 0; i < get_input_size(); i++)
        input_shapes.push_back(get_input_partial_shape(i));

    const auto result_shapes = op::v4::shape_infer(this, input_shapes);

    set_output_type(0, m_output_type, result_shapes[0]);
}

shared_ptr<Node> op::v4::Range::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Range_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v4::Range>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_type);
}

template <typename T>
bool get_casted_value(const HostTensorPtr& tensor, T* val) {
    switch (tensor->get_element_type()) {
    case element::Type_t::bf16:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::bf16>());
        break;
    case element::Type_t::f16:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::f16>());
        break;
    case element::Type_t::f32:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::f32>());
        break;
    case element::Type_t::i8:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::i8>());
        break;
    case element::Type_t::i32:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::i32>());
        break;
    case element::Type_t::i64:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::i64>());
        break;
    case element::Type_t::u8:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::u8>());
        break;
    case element::Type_t::u32:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::u32>());
        break;
    case element::Type_t::u64:
        *val = static_cast<T>(*tensor->get_data_ptr<element::Type_t::u64>());
        break;
    default:
        return false;
    }
    return true;
}

namespace rangeop {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& out,
              const HostTensorPtr& start,
              const HostTensorPtr& stop,
              const HostTensorPtr& step,
              int version) {
    using T = typename element_type_traits<ET>::value_type;
    T start_val;
    T stop_val;
    T step_val;
    if (version < 4) {
        start_val = *start->get_data_ptr<ET>();
        stop_val = *stop->get_data_ptr<ET>();
        step_val = *step->get_data_ptr<ET>();
        if (!(check_value(start_val) && check_value(stop_val) && check_value(step_val) &&
              (step_val != static_cast<T>(0)))) {
            return false;
        }
    } else {
        if (!(get_casted_value<T>(start, &start_val) && get_casted_value<T>(stop, &stop_val) &&
              get_casted_value<T>(step, &step_val))) {
            return false;
        }
    }

    int64_t out_size = 0;

    int64_t steps = static_cast<int64_t>(std::ceil(double(stop_val - start_val) / step_val));
    if (steps > 0) {
        out_size = steps;
    }
    ov::Shape out_shape = ov::Shape({static_cast<size_t>(out_size)});
    out->set_shape(out_shape);
    runtime::reference::range(&start_val, &step_val, shape_size(out_shape), out->get_data_ptr<ET>());
    return true;
}

bool evaluate_power(const HostTensorPtr& out,
                    const HostTensorPtr& start,
                    const HostTensorPtr& stop,
                    const HostTensorPtr& step,
                    const element::Type& output_type,
                    int version) {
    bool rc = true;
    switch (output_type) {
        NGRAPH_TYPE_CASE(evaluate_range, bf16, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, f16, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, f32, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, f64, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, i8, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, i16, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, i32, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, i64, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, u8, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, u16, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, u32, out, start, stop, step, version);
        NGRAPH_TYPE_CASE(evaluate_range, u64, out, start, stop, step, version);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace rangeop

bool op::v4::Range::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v4_Range_evaluate);
    HostTensorPtr out = outputs[0];
    HostTensorPtr start = inputs[0];
    HostTensorPtr stop = inputs[1];
    HostTensorPtr step = inputs[2];
    return rangeop::evaluate_power(out, start, stop, step, m_output_type, 4);
}

bool op::v4::Range::has_evaluate() const {
    OV_OP_SCOPE(v4_Range_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        return true;
    default:
        break;
    }
    return false;
}

op::v0::Range::Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step)
    : Op({start, stop, step}) {
    constructor_validate_and_infer_types();
}

template <typename T>
static void check_start(const op::v0::Range* node, T start) {
    NODE_VALIDATION_CHECK(node, check_value(start), "'start' cannot be nan or infinite.");
}

template <typename T>
void check_stop(const op::v0::Range* node, T stop) {
    NODE_VALIDATION_CHECK(node, check_value(stop), "'stop' cannot be nan or infinite.");
}

template <typename T>
void static check_step(const op::v0::Range* node, T step) {
    NODE_VALIDATION_CHECK(node,
                          check_value(step) && ((step > static_cast<T>(0) || step < static_cast<T>(0))),
                          "'step' cannot be zero, nan, or infinite.");
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, T>::type adjust_for_step_and_sign(T span, T step) {
    return ceil_div(span < 0 ? -static_cast<typename std::make_signed<T>::type>(span) : span,
                    step < 0 ? -static_cast<typename std::make_signed<T>::type>(step) : step);
}

template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                   std::is_same<T, bfloat16>::value,
                               T>::type
adjust_for_step_and_sign(T span, T step) {
    return ceil(fabs(span) / fabs(step));
}

template <typename T>
static ov::PartialShape infer_output_shape(const op::v0::Range* node, const element::Type& /* et */) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto const_start = get_constant_from_source(node->input_value(0));
    auto const_stop = get_constant_from_source(node->input_value(1));
    auto const_step = get_constant_from_source(node->input_value(2));
    OPENVINO_SUPPRESS_DEPRECATED_END

    T start = static_cast<T>(0);
    T stop = static_cast<T>(0);
    T step = static_cast<T>(0);

    if (const_start != nullptr) {
        std::vector<T> start_val = const_start->get_vector<T>();
        NODE_VALIDATION_CHECK(node, start_val.size() == 1);
        start = start_val[0];
        check_start<T>(node, start);
    }

    if (const_stop != nullptr) {
        std::vector<T> stop_val = const_stop->get_vector<T>();
        NODE_VALIDATION_CHECK(node, stop_val.size() == 1);
        stop = stop_val[0];
        check_stop<T>(node, stop);
    }

    if (const_step != nullptr) {
        std::vector<T> step_val = const_step->get_vector<T>();
        NODE_VALIDATION_CHECK(node, step_val.size() == 1);
        step = step_val[0];
        check_step<T>(node, step);
    }

    ov::PartialShape result{ov::PartialShape::dynamic(1)};

    if (const_start != nullptr && const_stop != nullptr && const_step != nullptr) {
        T span;

        if (step > static_cast<T>(0) && start >= stop) {
            span = static_cast<T>(0);
        } else if (step < static_cast<T>(0) && start <= stop) {
            span = static_cast<T>(0);
        } else {
            span = stop - start;
        }

        T strided = adjust_for_step_and_sign<T>(span, step);

        result = ov::PartialShape{Dimension(static_cast<int64_t>(strided))};
    }

    return result;
}

bool ngraph::op::v0::Range::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Range_visit_attributes);
    return true;
}

void op::v0::Range::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Range_validate_and_infer_types);
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);

    auto result_et = element::dynamic;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)),
                          "Element types for start, stop, and step do not match.");

    NODE_VALIDATION_CHECK(this,
                          result_et != element::boolean,
                          "Element type for start, stop, and step, must not be boolean.");

    NODE_VALIDATION_CHECK(this,
                          result_et != element::Type_t::u1 && result_et != element::Type_t::i4 &&
                              result_et != element::Type_t::u4 && result_et != element::Type_t::undefined,
                          "Internal OpenVINO error: unsupported element type: ",
                          result_et);

    if (result_et == element::Type_t::dynamic) {
        set_output_type(0, result_et, ov::PartialShape::dynamic(1));
    } else {
        std::vector<PartialShape> input_shapes;
        for (size_t i = 0; i < get_input_size(); i++)
            input_shapes.push_back(get_input_partial_shape(i));

        const auto result_shapes = op::v0::shape_infer(this, input_shapes);

        set_output_type(0, result_et, result_shapes[0]);
    }
}

shared_ptr<Node> op::v0::Range::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Range_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Range>(new_args.at(0), new_args.at(1), new_args.at(2));
}

template <element::Type_t ET, typename T>
void positive_range(T start_val, T stop_val, T step_val) {}

bool op::v0::Range::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Range_evaluate);
    HostTensorPtr out = outputs[0];
    HostTensorPtr start = inputs[0];
    HostTensorPtr stop = inputs[1];
    HostTensorPtr step = inputs[2];
    return rangeop::evaluate_power(out, start, stop, step, start->get_element_type(), 0);
}

bool op::v0::Range::has_evaluate() const {
    OV_OP_SCOPE(v0_Range_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        return true;
    default:
        break;
    }
    return false;
}
