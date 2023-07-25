// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

std::map<std::string, InputInfo> get_input_info_by_node(const std::shared_ptr<ov::Node>& node) {
    std::map<std::string, InputInfo> input_info;
    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        InputInfo in_info;
        std::shared_ptr<ov::Node> input_node = node->get_input_node_shared_ptr(port_id);
        std::string input_name = input_node->get_friendly_name();
        if (std::dynamic_pointer_cast<ov::op::v0::Constant>(input_node)) {
            auto const_node =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(input_node);
            in_info.is_const = true;
            switch (node->get_output_element_type(0)) {
            case ov::element::Type_t::boolean: {
                in_info.ranges = get_const_ranges<bool>(const_node);
                break;
            }
            case ov::element::Type_t::bf16: {
                in_info.ranges = get_const_ranges<ov::bfloat16>(const_node);
                break;
            }
            case ov::element::Type_t::f16: {
                in_info.ranges = get_const_ranges<ov::float16>(const_node);
                break;
            }
            case ov::element::Type_t::f32: {
                in_info.ranges = get_const_ranges<float>(const_node);
                break;
            }
            case ov::element::Type_t::f64: {
                in_info.ranges = get_const_ranges<double>(const_node);
                break;
            }
            case ov::element::Type_t::i8: {
                in_info.ranges = get_const_ranges<int8_t>(const_node);
                break;
            }
            case ov::element::Type_t::i16: {
                in_info.ranges = get_const_ranges<int16_t>(const_node);
                break;
            }
            case ov::element::Type_t::i32: {
                in_info.ranges = get_const_ranges<int32_t>(const_node);
                break;
            }
            case ov::element::Type_t::i64: {
                in_info.ranges = get_const_ranges<int64_t>(const_node);
                break;
            }
                // TODO cast_vector doesn't support u1 now
                //        case ov::element::Type_t::u1:
                //            return get_const_ranges<char>(const_node);
            case ov::element::Type_t::u8: {
                in_info.ranges = get_const_ranges<uint8_t>(const_node);
                break;
            }
            case ov::element::Type_t::u16: {
                in_info.ranges = get_const_ranges<uint16_t>(const_node);
                break;
            }
            case ov::element::Type_t::u32: {
                in_info.ranges = get_const_ranges<uint32_t>(const_node);
                break;
            }
            case ov::element::Type_t::u64: {
                in_info.ranges = get_const_ranges<uint64_t>(const_node);
                break;
            }
            default: {
                std::cout << "Can't get ranges.. Unsupported data type" << std::endl;
                break;
            }}
        }
        if (ov::op::util::is_parameter(input_node) || ov::op::util::is_constant(input_node)) {
            input_info.insert({ input_name, in_info });
        }
    }
    return input_info;
}

std::shared_ptr<ov::Node> clone_node(std::shared_ptr<ov::Node> node,
                                     bool is_save_const,
                                     bool is_copy_const_node,
                                     std::string node_name) {
    // pass::Manager pass_manager;
    // pass_manager.register_pass<pass::ConstantFolding>();
    // auto model = std::make_shared<ov::Model>(node);
    // pass_manager.run_passes(model, node);

    bool has_parameters = false;
    ov::OutputVector inputs;
    inputs.resize(node->get_input_size());
    if (node_name.empty()) {
        node_name = ov::test::functional::get_node_version(node);
    }
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        std::string input_name = node_name + "_" + std::to_string(i);
        // todo: replace deprecated code && remove this w/a for constant size
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto constant_input = ov::get_constant_from_source(node->input(i).get_source_output());
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (constant_input) {
            if (is_save_const || constant_input->get_byte_size() <= 1024) {
                auto in_const = std::make_shared<ov::op::v0::Constant>(constant_input->get_element_type(),
                                                                       constant_input->get_shape(),
                                                                       constant_input->get_data_ptr());
                in_const->set_friendly_name(input_name);
                inputs[i] = in_const;
                continue;
            }
        }
        has_parameters = true;
        auto param =
            std::make_shared<ov::op::v0::Parameter>(node->get_input_element_type(i), node->get_input_partial_shape(i));
        param->set_friendly_name(input_name);
        inputs[i] = param;
    }
    if (!has_parameters && !is_copy_const_node) {
        auto cloned_node = clone_node(node, true, true);
        std::cout << "The operation: " + node->get_friendly_name() + " does not have parameters! Replace first input to parameter!" << std::endl;
        auto param =
            std::make_shared<ov::op::v0::Parameter>(cloned_node->get_input_element_type(0), cloned_node->get_input_partial_shape(0));
        std::string param_name = node_name + "_0";
        param->set_friendly_name(param_name);
        auto node_to_replace = cloned_node->get_input_node_shared_ptr(0);
        ov::replace_node(node_to_replace, param);
        return cloned_node;
    }
    std::shared_ptr<ov::Node> cloned_node = node->clone_with_new_inputs(inputs);
    cloned_node->set_friendly_name(node_name);
    return cloned_node;
}

std::shared_ptr<ov::Model> generate_model_by_node(const std::shared_ptr<ov::Node>& node) {
    static size_t model_cnt = 0;
    auto cloned_node = clone_node(node);
    ov::OutputVector results;
    for (auto& out : cloned_node->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(out));
    }
    auto model = std::make_shared<ov::Model>(results);
    model->set_friendly_name(cloned_node->get_friendly_name() + "_" + std::to_string(model_cnt++));
    return model;
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov