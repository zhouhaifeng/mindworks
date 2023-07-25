// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/sigmoid.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/sigmoid.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> ov::op::v0::Sigmoid::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Sigmoid_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Sigmoid>(new_args.at(0));
}

ov::op::v0::Sigmoid::Sigmoid(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

namespace sigmoid {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::sigmoid<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_sigmoid(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(arg0->get_shape());
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_sigmoid, i32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_sigmoid, i64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_sigmoid, u32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_sigmoid, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_sigmoid, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_sigmoid, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace sigmoid

bool ov::op::v0::Sigmoid::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Sigmoid_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return sigmoid::evaluate_sigmoid(inputs[0], outputs[0]);
}

bool ov::op::v0::Sigmoid::has_evaluate() const {
    OV_OP_SCOPE(v0_Sigmoid_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
