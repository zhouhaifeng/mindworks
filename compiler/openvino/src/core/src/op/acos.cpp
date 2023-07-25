// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/acos.hpp"

#include <string>

#include "itt.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/acos.hpp"

ov::op::v0::Acos::Acos(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v0::Acos::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Acos_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Acos>(new_args.at(0));
}

namespace acosop {
namespace {
template <ov::element::Type_t ET>
inline bool evaluate(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out, const size_t count) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::acos<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_acos(const ov::HostTensorPtr& arg0, const ov::HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_acos, i32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_acos, i64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_acos, u32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_acos, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_acos, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_acos, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace acosop

bool ov::op::v0::Acos::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Acos_evaluate);
    return acosop::evaluate_acos(inputs[0], outputs[0], shape_size(inputs[0]->get_shape()));
}

bool ov::op::v0::Acos::has_evaluate() const {
    OV_OP_SCOPE(v0_Acos_has_evaluate);
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
