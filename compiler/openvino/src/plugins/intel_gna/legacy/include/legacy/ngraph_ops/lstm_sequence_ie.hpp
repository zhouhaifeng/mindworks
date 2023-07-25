// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/op.hpp"
#include "ngraph/opsets/opset4.hpp"

namespace ngraph {
namespace op {
class LSTMSequenceIE : public ngraph::op::util::RNNCellBase {
public:
    OPENVINO_OP("LSTMSequenceIE", "legacy");

    LSTMSequenceIE() = delete;

    LSTMSequenceIE(const Output<Node>& X,
                   const Output<Node>& H_t,
                   const Output<Node>& C_t,
                   const Output<Node>& seq_lengths,
                   const Output<Node>& WR,
                   const Output<Node>& B,
                   size_t hidden_size,
                   ngraph::op::RecurrentSequenceDirection lstm_direction,
                   const std::vector<std::string>& activations,
                   const std::vector<float>& activations_alpha,
                   const std::vector<float>& activations_beta,
                   float clip,
                   int64_t seq_len = 1);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    ngraph::op::RecurrentSequenceDirection get_direction() {
        return m_direction;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    ngraph::op::RecurrentSequenceDirection m_direction;
    int64_t m_seq_axis;
};
}  // namespace op
}  // namespace ngraph
