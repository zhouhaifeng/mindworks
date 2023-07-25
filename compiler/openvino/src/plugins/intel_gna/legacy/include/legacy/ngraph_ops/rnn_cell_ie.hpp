// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class RNNCellIE : public Op {
public:
    OPENVINO_OP("RNNCellIE", "legacy");

    RNNCellIE(const Output<Node>& X,
              const Output<Node>& H_t,
              const Output<Node>& WR,
              const Output<Node>& B,
              size_t hidden_size,
              const std::vector<std::string>& activations,
              const std::vector<float>& activations_alpha,
              const std::vector<float>& activations_beta,
              float clip);

    RNNCellIE() = delete;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    std::size_t get_hidden_size() {
        return m_hidden_size;
    }
    const std::vector<std::string>& get_activations() {
        return m_activations;
    }
    const std::vector<float>& get_activations_alpha() {
        return m_activations_alpha;
    }
    const std::vector<float>& get_activations_beta() {
        return m_activations_beta;
    }
    float get_clip() {
        return m_clip;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    int64_t m_hidden_size{};

    std::vector<std::string> m_activations;
    std::vector<float> m_activations_alpha;
    std::vector<float> m_activations_beta;
    float m_clip;
};

}  // namespace op
}  // namespace ngraph
