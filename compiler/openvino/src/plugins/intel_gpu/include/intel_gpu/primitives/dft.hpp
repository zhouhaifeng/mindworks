// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/shape.hpp>
#include <utility>

#include "primitive.hpp"

namespace cldnn {

/// @brief Direction of DFT operation.
enum class dft_direction {
    forward,
    inverse,
};

/// @brief Mode of DFT operation.
enum class dft_mode {
    complex,
    real,
};

/// @brief DFT primitive.
struct dft : public primitive_base<dft> {
    CLDNN_DECLARE_PRIMITIVE(dft)

    dft() : primitive_base("", {}) {}

    DECLARE_OBJECT_TYPE_SERIALIZATION

    /// @brief Constructs DFT primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axes Axes to perform DFT.
    /// @param signal_size Signal sizes for 'axes'.
    /// @param output_shape Output shape.
    /// @param direction Direction of DFT operation.
    /// @param mode Mode of DFT operation.
    dft(const primitive_id& id,
        const input_info& input,
        std::vector<int64_t> axes,
        std::vector<int64_t> signal_size,
        const ov::Shape& output_shape,
        dft_direction direction,
        dft_mode mode,
        const padding& output_padding = {})
        : primitive_base(id, {input}, {output_padding}),
          axes(std::move(axes)),
          signal_size(std::move(signal_size)),
          output_shape(output_shape),
          direction(direction),
          mode(mode) {}

    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    ov::Shape output_shape;
    dft_direction direction;
    dft_mode mode;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, axes.begin(), axes.end());
        seed = hash_range(seed, signal_size.begin(), signal_size.end());
        seed = hash_combine(seed, direction);
        seed = hash_combine(seed, mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const dft>(rhs);

        return axes == rhs_casted.axes &&
               signal_size == rhs_casted.signal_size &&
               direction == rhs_casted.direction &&
               mode == rhs_casted.mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<dft>::save(ob);
        ob << axes;
        ob << signal_size;
        ob << output_shape;
        ob << make_data(&direction, sizeof(dft_direction));
        ob << make_data(&mode, sizeof(dft_mode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<dft>::load(ib);
        ib >> axes;
        ib >> signal_size;
        ib >> output_shape;
        ib >> make_data(&direction, sizeof(dft_direction));
        ib >> make_data(&mode, sizeof(dft_mode));
    }
};

}  // namespace cldnn
