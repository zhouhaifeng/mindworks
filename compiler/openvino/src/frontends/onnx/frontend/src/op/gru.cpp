// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/gru.hpp"

#include <string>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/shape.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/recurrent.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
namespace {
struct GRUInputMap : public recurrent::OpInputMap {
    GRUInputMap(const Node& node, std::size_t gates_count) : OpInputMap(node, gates_count) {
        bool linear_before_reset = static_cast<bool>(node.get_attribute_value<std::int64_t>("linear_before_reset", 0));

        // Override bias, since we need separated W and R biases for `h` gate.
        if (linear_before_reset) {
            const auto& ng_inputs = node.get_ng_inputs();
            const auto el_type = ng_inputs.at(0).get_element_type();

            if (ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs.at(3))) {
                auto bias = ng_inputs.at(3);
                // gates_count * 2 since B is: [Wb, Rb]
                const int split_parts = 2 * 3;
                const auto split_bias = builder::opset1::split(bias, split_parts, 1);
                const auto wr_z_bias = std::make_shared<default_opset::Add>(split_bias.at(0), split_bias.at(3));
                const auto wr_r_bias = std::make_shared<default_opset::Add>(split_bias.at(1), split_bias.at(4));
                // The result has shape: [num_directions, 4 * hidden_size]
                // and data layout:
                //       [
                //          [Wb_z + Rb_z],
                //          [Wb_r + Rb_r],
                //          [Wb_h],
                //          [Rb_h],
                //          // num_directions times
                //       ]
                m_map[recurrent::OpInput::B] = std::make_shared<default_opset::Concat>(
                    OutputVector{wr_z_bias, wr_r_bias, split_bias.at(2), split_bias.at(5)},
                    1);
            } else {
                const std::size_t hidden_size = m_map[recurrent::OpInput::R].get_shape().back();
                const std::size_t num_directions = m_map[recurrent::OpInput::W].get_shape().front();

                m_map[recurrent::OpInput::B] =
                    std::make_shared<default_opset::Constant>(el_type,
                                                              Shape{num_directions, (gates_count + 1) * hidden_size},
                                                              0.f);
            }
        }
    }

    virtual ~GRUInputMap() = default;
};

struct GRUAttributes : public recurrent::OpAttributes {
    GRUAttributes(const Node& node)
        : OpAttributes(node),
          m_linear_before_reset{static_cast<bool>(node.get_attribute_value<std::int64_t>("linear_before_reset", 0))} {
        m_activations = node.get_attribute_value<std::vector<std::string>>("activations", {"sigmoid", "tanh"});
    }

    virtual ~GRUAttributes() = default;

    bool m_linear_before_reset;
};
}  // namespace

OutputVector gru(const Node& node) {
    constexpr std::size_t gates_count = 3;
    GRUInputMap input_map{node, gates_count};
    GRUAttributes attributes{node};

    auto gru_sequence = std::make_shared<default_opset::GRUSequence>(input_map.at(recurrent::OpInput::X),
                                                                     input_map.at(recurrent::OpInput::INIT_H),
                                                                     input_map.at(recurrent::OpInput::SEQ_LENGTHS),
                                                                     input_map.at(recurrent::OpInput::W),
                                                                     input_map.at(recurrent::OpInput::R),
                                                                     input_map.at(recurrent::OpInput::B),
                                                                     attributes.m_hidden_size,
                                                                     attributes.m_direction,
                                                                     attributes.m_activations,
                                                                     attributes.m_activations_alpha,
                                                                     attributes.m_activations_beta,
                                                                     attributes.m_clip_threshold,
                                                                     attributes.m_linear_before_reset);

    const auto Y = gru_sequence->output(0);
    const auto Y_h = gru_sequence->output(1);

    return {builder::opset1::reorder_axes(Y, {2, 1, 0, 3}), builder::opset1::reorder_axes(Y_h, {1, 0, 2})};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
