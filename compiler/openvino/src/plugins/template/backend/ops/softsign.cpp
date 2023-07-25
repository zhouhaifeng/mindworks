// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/softsign.hpp"

#include "evaluate_node.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v9::SoftSign>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    ngraph::element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case ngraph::element::Type_t::f64:
        ngraph::runtime::reference::softsign<double>(inputs[0]->get_data_ptr<double>(),
                                                     outputs[0]->get_data_ptr<double>(),
                                                     ngraph::shape_size(inputs[0]->get_shape()));
        break;
    case ngraph::element::Type_t::f32:
        ngraph::runtime::reference::softsign<float>(inputs[0]->get_data_ptr<float>(),
                                                    outputs[0]->get_data_ptr<float>(),
                                                    ngraph::shape_size(inputs[0]->get_shape()));
        break;
    case ngraph::element::Type_t::f16:
        ngraph::runtime::reference::softsign<ngraph::float16>(inputs[0]->get_data_ptr<ngraph::float16>(),
                                                              outputs[0]->get_data_ptr<ngraph::float16>(),
                                                              ngraph::shape_size(inputs[0]->get_shape()));
        break;
    case ngraph::element::Type_t::bf16:
        ngraph::runtime::reference::softsign<ngraph::bfloat16>(inputs[0]->get_data_ptr<ngraph::bfloat16>(),
                                                               outputs[0]->get_data_ptr<ngraph::bfloat16>(),
                                                               ngraph::shape_size(inputs[0]->get_shape()));
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v9::SoftSign>(std::shared_ptr<ngraph::Node> node,
                                             const ngraph::HostTensorVector& outputs,
                                             const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v9::SoftSign>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
