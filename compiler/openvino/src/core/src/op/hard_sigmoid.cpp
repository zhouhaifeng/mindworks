// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/hard_sigmoid.hpp"

#include <memory>

#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

op::v0::HardSigmoid::HardSigmoid() : Op() {}

op::v0::HardSigmoid::HardSigmoid(const Output<Node>& data, const Output<Node>& alpha, const Output<Node>& beta)
    : Op({data, alpha, beta}) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::HardSigmoid::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_HardSigmoid_visit_attributes);
    return true;
}

void op::v0::HardSigmoid::validate_and_infer_types() {
    OV_OP_SCOPE(v0_HardSigmoid_validate_and_infer_types);
    const auto& alpha_pshape = get_input_partial_shape(1);
    const auto& beta_pshape = get_input_partial_shape(2);

    if (alpha_pshape.is_static()) {
        const auto alpha_shape = alpha_pshape.to_shape();
        NODE_VALIDATION_CHECK(this,
                              ngraph::is_scalar(alpha_shape),
                              "A scalar is expected for the 'alpha' input. Got: ",
                              alpha_shape);
    }

    if (beta_pshape.is_static()) {
        const auto beta_shape = beta_pshape.to_shape();
        NODE_VALIDATION_CHECK(this,
                              ngraph::is_scalar(beta_shape),
                              "A scalar is expected for the 'beta' input. Got: ",
                              beta_shape);
    }

    const auto& data_et = get_input_element_type(0);
    const auto& alpha_et = get_input_element_type(1);
    const auto& beta_et = get_input_element_type(2);

    NODE_VALIDATION_CHECK(this,
                          data_et == alpha_et && data_et == beta_et,
                          "The element types of both alpha and beta inputs must match the data input type.");

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v0::HardSigmoid::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_HardSigmoid_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<op::v0::HardSigmoid>(new_args.at(0), new_args.at(1), new_args.at(2));
}
