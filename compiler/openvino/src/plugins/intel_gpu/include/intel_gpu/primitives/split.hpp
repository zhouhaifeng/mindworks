// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>
#include <utility>

namespace cldnn {

/// @brief Performs split operation on input.
/// @details splits the input data into n parts, for each user provides name and offsets.
/// @n User cannot use split primitive directly.
/// @n It is needed to refer to the output ids with the name "<split_prim_id>:<split_output_id>".
/// @n
/// @n\b Assumptions
/// @n - offsets1 < offsets2 < offsets3 < ...
/// @n - size[n] = offsets[n+1] - offsets[n];
/// @n - last element: size[n] = split_input.size - offsets[n];
/// @n - no buffer overlapping, as the output size is calculated using offset and input size
/// @n - split primitive id cannot be used by any other primitive (user needs to use output_ids only)
/// @n Breaking any of this conditions will cause exeption throw.
/// @n
/// @n\b Example:
/// @n Splitting output to 2 parts by the features:
/// @n input_size = { 2, 4, 3, 5 };
/// @n split_id = "split";
/// @n output_ids_offsets[0] = { "out0", { 0,0,0,0 } };
/// @n output_ids_offsets[1] = { "out1", { 0,2,0,0 } };
/// @n After split there would be 2 primitives: "split:out0" and "split:out1" which contain 2 feature maps (lower and upper)
struct split : public primitive_base<split> {
    CLDNN_DECLARE_PRIMITIVE(split)

    /// @brief Constructs split primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_ids_offsets Pairs of output_ids and offsets
    split(const primitive_id& id,
          const input_info& input,
          const std::vector<std::pair<primitive_id, tensor> >& output_ids_offsets,
          const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          output_offsets(extract_tensor_vector(output_ids_offsets)),
          output_ids(extract_primitive_vector(output_ids_offsets)) {}

    /// @brief Array of tensors with offsets.
    std::vector<tensor> output_offsets;
    /// @brief List of output_ids.
    const primitive_id_arr output_ids;

    size_t hash() const override {
        size_t seed = primitive::hash();
        for (auto& offset : output_offsets) {
            seed = hash_combine(seed, offset.hash());
        }
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const split>(rhs);

        return output_offsets == rhs_casted.output_offsets;
    }

protected:
    static std::vector<primitive_id> extract_primitive_vector(
        const std::vector<std::pair<primitive_id, tensor> >& stor) {
        std::vector<primitive_id> res;
        for (auto& stor_pair : stor) res.push_back(stor_pair.first);

        return res;
    }

    static std::vector<tensor> extract_tensor_vector(const std::vector<std::pair<primitive_id, tensor> >& stor) {
        std::vector<tensor> res;
        for (auto& stor_pair : stor) res.push_back(stor_pair.second);

        return res;
    }
};
}  // namespace cldnn
