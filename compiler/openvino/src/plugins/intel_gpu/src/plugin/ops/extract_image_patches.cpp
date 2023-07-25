// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/extractimagepatches.hpp"

#include "intel_gpu/primitives/extract_image_patches.hpp"

namespace ov {
namespace intel_gpu {

static inline std::string PadToString(ngraph::op::PadType pad) {
    switch (pad) {
        case ngraph::op::PadType::SAME_UPPER: return "same_upper";
        case ngraph::op::PadType::SAME_LOWER: return "same_lower";
        case ngraph::op::PadType::VALID: return "valid";
        default: OPENVINO_THROW("Unsupported pad type in ExtractImagePatches primitive ", pad);
    }

    return "";
}

static void CreateExtractImagePatchesOp(Program& p, const std::shared_ptr<ngraph::op::v3::ExtractImagePatches>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint32_t> sizes;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> rates;
    for (auto size : op->get_sizes()) {
        sizes.push_back(static_cast<uint32_t>(size));
    }
    for (auto stride : op->get_strides()) {
        strides.push_back(static_cast<uint32_t>(stride));
    }
    for (auto rate : op->get_rates()) {
        rates.push_back(static_cast<uint32_t>(rate));
    }
    std::string auto_pad = PadToString(op->get_auto_pad());

    auto extractImagePatchesPrim = cldnn::extract_image_patches(layerName,
                                                                inputs[0],
                                                                sizes,
                                                                strides,
                                                                rates,
                                                                auto_pad,
                                                                tensor_from_dims(op->get_output_shape(0)));

    p.add_primitive(*op, extractImagePatchesPrim);
}

REGISTER_FACTORY_IMPL(v3, ExtractImagePatches);

}  // namespace intel_gpu
}  // namespace ov
