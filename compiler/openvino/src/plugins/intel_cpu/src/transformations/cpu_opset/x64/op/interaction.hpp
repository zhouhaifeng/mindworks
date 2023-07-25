// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class InteractionNode : public ngraph::op::Op {
public:
    OPENVINO_OP("Interaction", "cpu_plugin_opset");

    InteractionNode() = default;

    InteractionNode(const OutputVector& args);

    InteractionNode(const NodeVector& args);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    ngraph::element::Type get_output_type() const { return m_output_type; }

    void set_fq_scales(const std::vector<float>& scales) {
        m_fq_scales = scales;
    }

    const std::vector<float>& get_output_scales() const {
        return m_fq_scales;
    }

private:
    ngraph::element::Type m_output_type;
    std::vector<float> m_fq_scales;
};

}   // namespace intel_cpu
}   // namespace ov
