// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

#pragma once

namespace ov {
namespace frontend {
namespace pytorch {
class PtFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("PtFrameworkNode", "util", ::ov::op::util::FrameworkNode);
    static constexpr const char* op_type_key = "PtTypeName";
    static constexpr const char* schema_key = "PtSchema";
    static constexpr const char* failed_conversion_key = "PtException";

    PtFrameworkNode(const std::shared_ptr<TorchDecoder>& decoder,
                    const OutputVector& inputs,
                    size_t output_size,
                    bool is_backprop = false)
        : ov::op::util::FrameworkNode(inputs, output_size, decoder->get_subgraph_size()),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name("PTFrameworkNode");
        if (is_backprop) {
            attrs[op_type_key] = m_decoder->get_op_type() + "_backprop";
            attrs[schema_key] = "None";
        } else {
            attrs[op_type_key] = m_decoder->get_op_type();
            attrs[schema_key] = m_decoder->get_schema();
        }
        set_attrs(attrs);

        // Set output shapes and types if recognized
        for (size_t i = 0; i < output_size; ++i) {
            PartialShape ps;
            // TODO: Try to decode PT type as a custom type
            auto type = element::dynamic;
            if (i < decoder->num_of_outputs()) {
                try {
                    ps = m_decoder->get_output_shape(i);
                } catch (...) {
                    // nothing, means the info cannot be queried and remains unknown
                }
            }
            // TODO: Set custom `type` via special API
            set_output_type(i, type, ps);
        }
    }

    PtFrameworkNode(const std::shared_ptr<TorchDecoder>& decoder, const OutputVector& inputs)
        : PtFrameworkNode(decoder, inputs, decoder->num_of_outputs()) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto op = std::make_shared<PtFrameworkNode>(m_decoder, inputs, get_output_size());

        for (size_t body_index = 0; body_index < m_bodies.size(); ++body_index) {
            op->set_function(static_cast<int>(body_index), get_function(static_cast<int>(body_index))->clone());
            for (const auto& m_input_descr : m_input_descriptions[body_index]) {
                op->m_input_descriptions[body_index].push_back(m_input_descr->copy());
            }
            for (const auto& m_output_descr : m_output_descriptions[body_index]) {
                op->m_output_descriptions[body_index].push_back(m_output_descr->copy());
            }
        }
        op->validate_and_infer_types();

        return op;
    }

    std::string get_op_type() const {
        return m_decoder->get_op_type();
    }

    TorchDecoder* get_decoder() const {
        return m_decoder.get();
    }

private:
    std::shared_ptr<TorchDecoder> m_decoder;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
