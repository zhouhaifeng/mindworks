// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reduce_logical_and.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/evaluate_helpers.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/logical_reduction.hpp"
#include "openvino/util/log.hpp"

using namespace ngraph;
using namespace std;

op::v1::ReduceLogicalAnd::ReduceLogicalAnd(const Output<Node>& data,
                                           const Output<Node>& reduction_axes,
                                           const bool keep_dims)
    : LogicalReductionKeepDims(data, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceLogicalAnd::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ReduceLogicalAnd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::ReduceLogicalAnd>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace reduce_and {
namespace {
bool evaluate_reduce_logical_and(const HostTensorPtr& data,
                                 const HostTensorPtr& out,
                                 const AxisSet& reduction_axes,
                                 bool keep_dims) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    out->set_shape(reduce(data->get_shape(), reduction_axes, keep_dims));
    try {
        runtime::reference::reduce_logical_and(data->get_data_ptr<char>(),
                                               out->get_data_ptr<char>(),
                                               data->get_shape(),
                                               reduction_axes);
        return true;
    } catch (const ngraph_error& e) {
        OPENVINO_WARN << e.what();
        return false;
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}
}  // namespace
}  // namespace reduce_and

bool op::v1::ReduceLogicalAnd::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_ReduceLogicalAnd_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));

    const auto& data = inputs[0];
    const auto& axes = inputs[1];
    const auto& out = outputs[0];
    if (data->get_element_type() != element::boolean || !axes->get_element_type().is_integral_number()) {
        return false;
    }
    const auto reduction_axes =
        get_normalized_axes_from_tensor(axes, data->get_partial_shape().rank(), get_friendly_name());
    OPENVINO_SUPPRESS_DEPRECATED_END
    return reduce_and::evaluate_reduce_logical_and(data, out, reduction_axes, get_keep_dims());
}

bool op::v1::ReduceLogicalAnd::has_evaluate() const {
    OV_OP_SCOPE(v1_ReduceLogicalAnd_has_evaluate);
    return get_input_element_type(0) == element::boolean && get_input_element_type(1).is_integral_number();
}
