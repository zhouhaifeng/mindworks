// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_pooling.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "roi_pooling_shape_inference.hpp"

using namespace std;

namespace ov {
namespace op {
namespace v0 {
ROIPooling::ROIPooling(const Output<Node>& input,
                       const Output<Node>& coords,
                       const ov::Shape& output_size,
                       const float spatial_scale,
                       const string& method)
    : Op({input, coords}),
      m_output_size(output_size),
      m_spatial_scale(spatial_scale),
      m_method(method) {
    constructor_validate_and_infer_types();
}

void ROIPooling::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ROIPooling_validate_and_infer_types);
    const auto& feat_maps_et = get_input_element_type(0);
    const auto& coords_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          feat_maps_et.is_real() && coords_et.is_real(),
                          "The data type for input and ROIs is expected to be a floating point type. Got: ",
                          feat_maps_et,
                          " and: ",
                          coords_et);

    NODE_VALIDATION_CHECK(this,
                          feat_maps_et == coords_et,
                          "Type of feature maps (inputs) and ROIs is expected to be the same. Got: ",
                          feat_maps_et,
                          " and: ",
                          coords_et);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shapes = shape_infer(this, get_node_input_partial_shapes(*this));
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, feat_maps_et, output_shapes[0]);

    const auto& feat_maps_ps = get_input_partial_shape(0);
    const auto& coords_ps = get_input_partial_shape(1);

    // if channel dimension, C, not known
    // feature maps input is used by shape specialization pass
    if (feat_maps_ps.rank().is_static() && feat_maps_ps[1].is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }

    // if number of ROIs, NUM_ROIS, not known
    // coordinate input is used by shape specialization pass
    if (coords_ps.rank().is_static() && coords_ps[0].is_dynamic()) {
        set_input_is_relevant_to_shape(1);
    }
}

shared_ptr<Node> ROIPooling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ROIPooling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ROIPooling>(new_args.at(0), new_args.at(1), m_output_size, m_spatial_scale, m_method);
}

bool ROIPooling::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ROIPooling_visit_attributes);
    visitor.on_attribute("output_size", m_output_size);
    visitor.on_attribute("pooled_h", m_output_size[0]);
    visitor.on_attribute("pooled_w", m_output_size[1]);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("method", m_method);
    return true;
}

void ROIPooling::set_output_roi(Shape output_size) {
    m_output_size = std::move(output_size);
}
const Shape& ROIPooling::get_output_roi() const {
    return m_output_size;
}

void ROIPooling::set_spatial_scale(float scale) {
    m_spatial_scale = scale;
}

void ROIPooling::set_method(std::string method_name) {
    m_method = std::move(method_name);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
