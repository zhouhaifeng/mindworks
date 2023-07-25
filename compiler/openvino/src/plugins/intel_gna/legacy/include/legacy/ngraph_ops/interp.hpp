// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <limits>
#include <memory>
#include <string>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

struct InterpolateIEAttrs {
    int height = -1;
    int width = -1;
    float zoom_factor = 0;
    float shrink_factor = 0;
    float scale_factor = 1.0;
    bool align_corners = true;
    bool antialias = true;
    std::string mode = "";
    int pad_beg = 0;
    int pad_end = 0;
};

class Interp : public Op {
public:
    OPENVINO_OP("Interp", "legacy");

    Interp(const Output<Node>& image, const InterpolateIEAttrs& attrs);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    InterpolateIEAttrs get_attrs() {
        return m_attrs;
    }

private:
    InterpolateIEAttrs m_attrs;
};

struct ResampleIEAttrs {
    bool antialias = true;
    int64_t factor = 0;
    std::string mode = "";
};

class ResampleV2 : public Op {
public:
    OPENVINO_OP("ResampleV2", "legacy");

    ResampleV2(const Output<Node>& image, const Output<Node>& output_shape, const ResampleIEAttrs& attrs);

    ResampleV2(const Output<Node>& image, const ResampleIEAttrs& attrs);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    ResampleIEAttrs get_attrs() {
        return m_attrs;
    }

private:
    ResampleIEAttrs m_attrs;
};

}  // namespace op
}  // namespace ngraph
