// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Bucketize operation bucketizes the input based on boundaries.
struct bucketize : primitive_base<bucketize> {
    CLDNN_DECLARE_PRIMITIVE(bucketize)

    bucketize() : primitive_base("", {}) {}

    DECLARE_OBJECT_TYPE_SERIALIZATION

    /// @brief Constructs bucketize primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param output_type Output tensor type.
    /// @param with_right_bound Indicates whether bucket includes the right or the left edge of interval.
    bucketize(const primitive_id& id,
              const std::vector<input_info>& inputs,
              data_types output_type = data_types::i64,
              bool with_right_bound = true,
              const padding& output_padding = {})
        : primitive_base(id, inputs, {output_padding}, {optional_data_type(output_type)}),
          with_right_bound(with_right_bound) {}

    bool with_right_bound;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, with_right_bound);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const bucketize>(rhs);

        return with_right_bound == rhs_casted.with_right_bound;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<bucketize>::save(ob);
        ob << with_right_bound;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<bucketize>::load(ib);
        ib >> with_right_bound;
    }
};

}  // namespace cldnn
