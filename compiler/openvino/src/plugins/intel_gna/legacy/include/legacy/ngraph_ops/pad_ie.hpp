// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/pad.hpp"

namespace ngraph {
namespace op {

class PadIE : public Op {
public:
    OPENVINO_OP("PadIE", "legacy");

    explicit PadIE(const std::shared_ptr<op::v1::Pad>& pad);

    PadIE(const Output<ngraph::Node>& input,
          PadMode pad_mode,
          CoordinateDiff pads_begin,
          CoordinateDiff pads_end,
          Shape output_shape,
          float pad_value);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    PadMode get_pad_mode() {
        return m_pad_mode;
    }
    CoordinateDiff get_pads_begin() {
        return m_pads_begin;
    }
    CoordinateDiff get_pads_end() {
        return m_pads_end;
    }
    float get_pad_value() {
        return m_pad_value;
    }

private:
    PadMode m_pad_mode;
    CoordinateDiff m_pads_begin, m_pads_end;
    Shape m_output_shape;
    float m_pad_value = 0;
};
}  // namespace op
}  // namespace ngraph
