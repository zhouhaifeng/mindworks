// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"

namespace cldnn {



/// @brief ExperimentalDetectronTopKROIs-6 primitive
/// @details
struct experimental_detectron_topk_rois : public primitive_base<experimental_detectron_topk_rois> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_topk_rois)

    experimental_detectron_topk_rois() : primitive_base("", {}) {}

    DECLARE_OBJECT_TYPE_SERIALIZATION

    /**
     * Construct ExperimentalDetectronTopKROIs privitive.
     * @param id primitive id
     * @param inputs inputs parameters ids
     * @param max_rois maximal numbers of output ROIs.
     */
    experimental_detectron_topk_rois(const primitive_id &id, const std::vector<input_info> &inputs,
                                     const size_t max_rois,
                                     const padding &output_padding = padding())
            : primitive_base(id, inputs, {output_padding}),
              max_rois(max_rois) {}

    /// maximal numbers of output ROIs.
    size_t max_rois;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, max_rois);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const experimental_detectron_topk_rois>(rhs);

        return max_rois == rhs_casted.max_rois;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<experimental_detectron_topk_rois>::save(ob);
        ob << max_rois;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<experimental_detectron_topk_rois>::load(ib);
        ib >> max_rois;
    }
};

}  // namespace cldnn
