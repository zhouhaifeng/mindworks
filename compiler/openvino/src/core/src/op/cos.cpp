// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/cos.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/cos.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::Cos::Cos(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool op::Cos::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Cos_visit_attributes);
    return true;
}

shared_ptr<Node> op::Cos::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Cos_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Cos>(new_args.at(0));
}

namespace cosop {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::cos<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_cos(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_cos, i32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_cos, i64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_cos, u32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_cos, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_cos, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_cos, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace cosop

bool op::Cos::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Cos_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return cosop::evaluate_cos(inputs[0], outputs[0], shape_size(inputs[0]->get_shape()));
}

bool op::Cos::has_evaluate() const {
    OV_OP_SCOPE(v0_Cos_has_evaluate);
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
