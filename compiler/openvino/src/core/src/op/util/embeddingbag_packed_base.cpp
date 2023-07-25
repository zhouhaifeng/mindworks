// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/embeddingbag_packed_base.hpp"

#include "embeddingbag_packed_shape_inference.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "openvino/core/validation_util.hpp"

using namespace std;

ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(const Output<Node>& emb_table,
                                                             const Output<Node>& indices,
                                                             const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, per_sample_weights}) {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(const Output<Node>& emb_table, const Output<Node>& indices)
    : Op({emb_table, indices}) {
    constructor_validate_and_infer_types();
}

void ov::op::util::EmbeddingBagPackedBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_EmbeddingBagPackedBase_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INDICES) == element::i64 || get_input_element_type(INDICES) == element::i32,
        "INDICES type must be i32 or i64");

    if (get_input_size() == 3) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(EMB_TABLE).compatible(get_input_element_type(PER_SAMPLE_WEIGHTS)),
                              "Per sample weight element type (",
                              get_input_element_type(PER_SAMPLE_WEIGHTS),
                              ") must match embedding table element type (",
                              get_input_element_type(EMB_TABLE),
                              ")");
    }

    const auto& emb_et = get_input_element_type(EMB_TABLE);
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, emb_et, shape_infer(this, input_shapes)[0]);
}

bool ov::op::util::EmbeddingBagPackedBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_EmbeddingBagPackedBase_visit_attributes);
    return true;
}
