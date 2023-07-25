// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/common.hpp"

#include <onnx/onnx_pb.h>  // onnx types

#include "default_opset.hpp"
#include "ngraph/graph_util.hpp"
#include "onnx_framework_node.hpp"
#include "openvino/core/deprecated.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace common {
const ngraph::element::Type& get_ngraph_element_type(int64_t onnx_type) {
    switch (onnx_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
        return element::boolean;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
        return element::f64;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        return element::f16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        return element::f32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        return element::i8;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
        return element::i16;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        return element::i32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        return element::i64;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        return element::u8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
        return element::u16;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        return element::u32;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
        return element::u64;
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
        return element::dynamic;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
        return element::bf16;
    }
    OPENVINO_THROW("unsupported element type");
}

std::shared_ptr<ngraph::Node> get_monotonic_range_along_node_rank(const Output<ngraph::Node>& value,
                                                                  int64_t start_value,
                                                                  int64_t step) {
    if (value.get_partial_shape().rank().is_static()) {
        const auto range_value =
            get_monotonic_range<int64_t>(value.get_partial_shape().rank().get_length(), start_value, step);
        return default_opset::Constant::create(element::i64, {range_value.size()}, range_value);
    }

    const auto value_shape = std::make_shared<default_opset::ShapeOf>(value);
    return std::make_shared<default_opset::Range>(default_opset::Constant::create(element::i64, {}, {start_value}),
                                                  std::make_shared<default_opset::ShapeOf>(value_shape),
                                                  default_opset::Constant::create(element::i64, {}, {step}),
                                                  element::i64);
}

void validate_scalar_input(const char* input_name,
                           const std::shared_ptr<ngraph::Node> input,
                           const std::set<element::Type> allowed_types) {
    const auto validated_input_shape = input->get_output_partial_shape(0);
    const auto validated_input_rank = validated_input_shape.rank();

    NGRAPH_CHECK(validated_input_rank.same_scheme({0}) ||
                     (validated_input_rank.same_scheme({1}) && validated_input_shape[0].get_length() == 1),
                 input_name,
                 " needs to be a scalar or 1D, single-element tensor.");

    if (!allowed_types.empty()) {
        const bool data_type_ok = allowed_types.count(input->get_element_type());
        NGRAPH_CHECK(data_type_ok, "Incorrect data type of the ", input_name, " input: ", input->get_element_type());
    }
}

template <typename T>
OutputVector handle_opset6_binary_op(const Node& node) {
    const Output<ngraph::Node> lhs_node = node.get_ng_inputs().at(0);
    Output<ngraph::Node> rhs_node = node.get_ng_inputs().at(1);
    const bool broadcast = node.get_attribute_value<std::int64_t>("broadcast", 0);
    if (broadcast) {
        if (node.has_attribute("axis")) {
            NGRAPH_CHECK(
                lhs_node.get_partial_shape().rank().is_static() && rhs_node.get_partial_shape().rank().is_static(),
                "Input's rank has to be static.");
            auto axis = node.get_attribute_value<std::int64_t>("axis");
            auto lhs_rank = lhs_node.get_partial_shape().rank().get_length();
            auto rhs_rank = rhs_node.get_partial_shape().rank().get_length();
            if (axis < 0)
                axis += lhs_rank;
            if (lhs_rank > axis + rhs_rank) {
                auto ones = default_opset::Constant::create(element::i64,
                                                            Shape{static_cast<size_t>(lhs_rank - axis - rhs_rank)},
                                                            std::vector<int64_t>(lhs_rank - axis - rhs_rank, 1));
                auto rhs_shape = std::make_shared<default_opset::ShapeOf>(rhs_node);
                auto new_shape = std::make_shared<default_opset::Concat>(OutputVector{rhs_shape, ones}, 0);
                rhs_node = std::make_shared<default_opset::Reshape>(rhs_node, new_shape, false);
            }
        } else {
            rhs_node = std::make_shared<default_opset::Broadcast>(rhs_node,
                                                                  std::make_shared<default_opset::ShapeOf>(lhs_node));
        }
    }
    return {std::make_shared<T>(lhs_node, rhs_node)};
}

template OutputVector handle_opset6_binary_op<default_opset::Add>(const Node& node);
template OutputVector handle_opset6_binary_op<default_opset::Divide>(const Node& node);
template OutputVector handle_opset6_binary_op<default_opset::Multiply>(const Node& node);
template OutputVector handle_opset6_binary_op<default_opset::Subtract>(const Node& node);
template OutputVector handle_opset6_binary_op<default_opset::LogicalAnd>(const Node& node);

const std::string FAILSAFE_NODE = "ONNX_FAILSAFE_NODE";

std::shared_ptr<default_opset::Constant> make_failsafe_constant(const ngraph::element::Type& dtype) {
    const auto failsafe_constant = default_opset::Constant::create(dtype, Shape{}, {0});
    auto& rt_info = failsafe_constant->get_rt_info();
    rt_info[FAILSAFE_NODE] = true;
    return failsafe_constant;
}

bool is_failsafe_node(const std::shared_ptr<ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(FAILSAFE_NODE) != rt_info.end();
}

const std::string OPTIMIZED_OUT_NODE = "OPTIMIZED_OUT_NODE";

void mark_as_optimized_out(Output<ov::Node>& node_output) {
    node_output.get_rt_info()[OPTIMIZED_OUT_NODE] = true;
}

bool is_optimized_out(const Output<ov::Node>& node_output) {
    const auto& rt_info = node_output.get_rt_info();
    return rt_info.find(OPTIMIZED_OUT_NODE) != rt_info.end();
}

std::string collect_translation_exceptions(const std::shared_ptr<ov::Model>& partially_converted) {
    std::string fully_unsupported_ops = "OpenVINO does not support the following ONNX operations: ";
    std::string additional_error_message = "Errors during ONNX translation: \n";
    const std::string sep = ", ";

    bool unsupported_found = false;
    bool additional_error_found = false;
    for (const auto& op : partially_converted->get_ops()) {
        if (const auto unsupported = std::dynamic_pointer_cast<frontend::NotSupportedONNXNode>(op)) {
            if (unsupported->additional_error_message().empty()) {
                fully_unsupported_ops += (unsupported->get_attrs().get_opset_name().empty()
                                              ? ""
                                              : unsupported->get_attrs().get_opset_name() + ".") +
                                         unsupported->get_attrs().get_type_name() + sep;
                unsupported_found = true;
            } else {
                additional_error_message += unsupported->additional_error_message();
                additional_error_found = true;
            }
        }
    }
    fully_unsupported_ops = fully_unsupported_ops.substr(0, fully_unsupported_ops.size() - sep.size());
    // remove redundant new line
    additional_error_message =
        (additional_error_message.empty() || additional_error_message[additional_error_message.length() - 1] != '\n')
            ? additional_error_message
            : additional_error_message.erase(additional_error_message.length() - 1);
    if (unsupported_found && additional_error_found) {
        return fully_unsupported_ops + "\n" + additional_error_message;
    } else if (unsupported_found) {
        return fully_unsupported_ops;
    } else if (additional_error_found) {
        return additional_error_message;
    } else {
        return "";
    }
}

}  // namespace  common
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
