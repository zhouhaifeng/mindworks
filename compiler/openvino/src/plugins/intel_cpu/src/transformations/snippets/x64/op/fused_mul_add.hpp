// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @interface FusedMulAdd
 * @brief Fused Multiply Add
 * @ingroup snippets
 */
class FusedMulAdd : public ngraph::op::Op {
public:
    OPENVINO_OP("FusedMulAdd", "SnippetsOpset");

    FusedMulAdd() = default;
    FusedMulAdd(const Output<Node>& a, const Output<Node>& b, const Output<Node>& c);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

} // namespace intel_cpu
} // namespace ov
