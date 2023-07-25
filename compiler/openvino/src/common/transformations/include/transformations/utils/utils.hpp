// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <functional>
#include <limits>
#include <memory>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations/rt_info/attributes.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace op {
namespace util {

template <class T>
bool normalize_single_value(std::vector<T> vec, float& value) {
    for (const auto& val : vec) {
        if (val != *vec.begin())
            return false;
    }

    float ref_val = static_cast<float>(*vec.begin());

    if (ref_val < std::numeric_limits<float>::lowest() || ref_val > std::numeric_limits<float>::max()) {
        return false;
    }

    value = ref_val;
    return true;
}

template <class T>
bool has_op_with_type(const std::shared_ptr<const ov::Model>& function) {
    for (const auto& op : function->get_ops()) {
        if (std::dynamic_pointer_cast<T>(op)) {
            return true;
        }
    }
    return false;
}

inline bool has_decompression_converts(const std::shared_ptr<const ov::Model>& function) {
    for (const auto& op : function->get_ops()) {
        if (std::dynamic_pointer_cast<opset8::Convert>(op)) {
            if (ov::is_decompression(op))
                return true;
        }
    }
    return false;
}

inline std::string create_ie_output_name(const Output<const Node>& output) {
    std::string out_name;
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto tensor_name = ov::descriptor::get_ov_tensor_legacy_name(output.get_tensor());
    NGRAPH_SUPPRESS_DEPRECATED_END
    if (!tensor_name.empty()) {
        out_name = std::move(tensor_name);
    } else {
        const auto& prev_layer = output.get_node_shared_ptr();
        out_name = prev_layer->get_friendly_name();
        if (prev_layer->get_output_size() != 1) {
            out_name += "." + std::to_string(output.get_index());
        }
    }
    return out_name;
}

inline std::string create_ie_output_name(const Output<Node>& output) {
    return create_ie_output_name(ov::Output<const Node>(output.get_node(), output.get_index()));
}

inline std::string get_ie_output_name(const Output<const Node>& output) {
    return create_ie_output_name(output);
}

inline std::string get_ie_output_name(const Output<Node>& output) {
    return get_ie_output_name(ov::Output<const Node>(output.get_node(), output.get_index()));
}

/**
 * \brief Convert epsilon value from double to float type.
 *
 * If the value is too large, the epsilon is converted to std::numeric_limits<float>::min() or
 * std::numeric_limits<float>::min(), otherwise static cast to float is called.
 * The adjustment is made for positive values only, for negative it works as static cast.
 *
 * \param eps  Original value of the epsilon (double).
 *
 * \return Epsilon value as float.
 */
float cast_eps_to_float(double eps_d);

template <typename T>
bool has_constant_value(const std::shared_ptr<Node>& node,
                        const T value,
                        T epsilon = std::numeric_limits<T>::epsilon()) {
    if (!node) {
        return false;
    }

    auto constant = std::dynamic_pointer_cast<opset4::Constant>(node);
    if (!constant) {
        return false;
    }

    const bool is_scalar_or_single_elem = is_scalar(constant->get_shape()) || shape_size(constant->get_shape()) == 1;
    if (!is_scalar_or_single_elem) {
        return false;
    }

    if (constant->get_element_type() == element::f16 || constant->get_element_type() == element::f32 ||
        constant->get_element_type() == element::f64 || constant->get_element_type() == element::bf16) {
        const auto data = constant->cast_vector<T>();
        if (std::fabs(data[0] - value) > epsilon) {
            return false;
        }
    } else {
        const auto data = constant->cast_vector<T>();
        if (data[0] != value) {
            return false;
        }
    }

    return true;
}

template <typename T>
bool has_constant_value(const std::shared_ptr<Node>& node,
                        const std::vector<T> values,
                        T epsilon = std::numeric_limits<T>::epsilon()) {
    if (!node) {
        return false;
    }

    auto constant = std::dynamic_pointer_cast<opset4::Constant>(node);
    if (!constant) {
        return false;
    }

    const auto const_values = constant->cast_vector<T>();

    if (constant->get_element_type() == element::f16 || constant->get_element_type() == element::f32 ||
        constant->get_element_type() == element::f64 || constant->get_element_type() == element::bf16) {
        return std::equal(const_values.cbegin(), const_values.cend(), values.cbegin(), [&](T lhs, T rhs) {
            return std::fabs(lhs - rhs) < epsilon;
        });
    }

    return const_values == values;
}

TRANSFORMATIONS_API bool get_single_value(const std::shared_ptr<opset4::Constant>& const_node, float& value);

TRANSFORMATIONS_API std::shared_ptr<Node> normalize_constant(const std::shared_ptr<opset4::Constant>& constant,
                                                             const PartialShape& shape);

TRANSFORMATIONS_API std::shared_ptr<Node> broadcastTo(const Output<Node>& input, const Shape& shape);

TRANSFORMATIONS_API std::shared_ptr<Node> reshapeTo(const Output<Node>& input, const Shape& shape);

TRANSFORMATIONS_API bool constantIsEqualTo(const std::shared_ptr<opset4::Constant>& const_node,
                                           float value,
                                           float eps = 1e-5);

TRANSFORMATIONS_API bool has_f16_constants(const std::shared_ptr<const ov::Model>& function);

TRANSFORMATIONS_API bool check_for_broadcast(const PartialShape& ref_shape, const PartialShape& other_shape);

TRANSFORMATIONS_API std::shared_ptr<Node> activation(const std::string& activation_name, const Output<Node>& apply_to);

TRANSFORMATIONS_API bool is_seq_len_provided(const std::shared_ptr<Node>& seq_len_input, int64_t max_seq_len);

TRANSFORMATIONS_API std::shared_ptr<Node> try_fold_unary_output(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API std::shared_ptr<Node> clone_try_fold(const std::shared_ptr<Node>& node, const OutputVector& inputs);

TRANSFORMATIONS_API bool shapes_equal_except_dynamic_expected_batch(const PartialShape& expected,
                                                                    const PartialShape& actual);

TRANSFORMATIONS_API void visit_shape_path(ov::Node* node,
                                          std::unordered_set<ov::Node*>& visited,
                                          std::function<void(ov::Node*)> func);

template <typename T, typename... Args>
std::shared_ptr<Node> make_try_fold(Args&&... args) {
    auto unary_output_node = std::make_shared<T>(std::forward<Args>(args)...);
    return try_fold_unary_output(unary_output_node);
}

template <class T>
Output<Node> eltwise_fold(const Output<Node>& input0, const Output<Node>& input1) {
    auto eltwise = std::make_shared<T>(input0, input1);
    OutputVector output(eltwise->get_output_size());
    OPENVINO_ASSERT(eltwise->constant_fold(output, {input0, input1}), "Can not constant fold eltwise node");
    OPENVINO_ASSERT(output.size() == 1, "Eltwise constant fold has unexpected number of outputs: ", output.size());
    return output[0];
}

TRANSFORMATIONS_API std::vector<Input<Node>> get_node_target_inputs(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API std::shared_ptr<Node> node_to_get_shape_value_of_indices_from_shape_node(
    const std::shared_ptr<Node>& shape_node,
    const std::vector<size_t>& indices);

TRANSFORMATIONS_API std::shared_ptr<Node> node_to_get_shape_value_of_indices_from_shape_source(
    const Output<Node>& shape_source,
    const std::vector<size_t>& indices);

TRANSFORMATIONS_API bool is_dequantization_subgraph(const Output<Node>& node);

TRANSFORMATIONS_API bool can_eliminate_eltwise_node(const std::shared_ptr<Node>& eltwise,
                                                    const Output<Node>& constant,
                                                    const Output<Node>& non_constant_input);

TRANSFORMATIONS_API bool is_constant_and_all_values_equal_int(const Output<Node>& output, const int64_t& v);

TRANSFORMATIONS_API bool is_on_constant_path(const ov::Output<ov::Node>& output);

}  // namespace util
}  // namespace op
}  // namespace ov
