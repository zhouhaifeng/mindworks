// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/itensor.hpp"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

ITensor::~ITensor() = default;

size_t ITensor::get_size() const {
    return shape_size(get_shape());
}

size_t ITensor::get_byte_size() const {
    return (get_size() * get_element_type().bitwidth() + 8 - 1) / 8;
}

bool ITensor::is_continuous() const {
    if (get_element_type().bitwidth() < 8)
        // OpenVINO doesn't support strides for lp types
        return true;
    const auto& shape = get_shape();
    const auto& type = get_element_type();
    std::vector<size_t> strides(shape.size());
    if (!shape.empty()) {
        strides[shape.size() - 1] = 1;
    }
    auto size = shape.size();
    for (size_t i = 1; i < size; i++) {
        strides[size - i - 1] = strides[size - i] * shape[size - i];
    }

    ov::Strides byte_strides(strides.size());
    for (size_t i = 0; i < strides.size(); ++i)
        byte_strides[i] = strides[i] * type.size();
    return byte_strides == get_strides();
}

void ITensor::copy_to(const std::shared_ptr<ov::ITensor>& dst) const {
    const auto& is_scalar = [](const ov::Shape& shape) {
        return shape.empty() || (shape.size() == 1 && shape[0] == 1);
    };
    const auto shapes_equal = [is_scalar](const ov::Shape& src, const ov::Shape& dst) {
        // WA for scalar tensors to copy {1} to {} or otherwise
        return src == dst || (is_scalar(src) && is_scalar(dst));
    };
    OPENVINO_ASSERT(dst, "Destination tensor was not initialized.");
    OPENVINO_ASSERT(!dynamic_cast<const ov::IRemoteTensor*>(this),
                    "Default copy to doesn't support copy from remote tensor.");
    OPENVINO_ASSERT(!std::dynamic_pointer_cast<ov::IRemoteTensor>(dst),
                    "Default copy to doesn't support copy to remote tensor.");
    OPENVINO_ASSERT(dst->get_element_type() == get_element_type(),
                    "Tensor element types are not equal. (src: ",
                    get_element_type(),
                    " != dst: ",
                    dst->get_element_type(),
                    ")");
    if (dst->get_shape() == ov::Shape{0})
        dst->set_shape(get_shape());
    OPENVINO_ASSERT(shapes_equal(get_shape(), dst->get_shape()),
                    "Tensor shapes are not equal. (src: ",
                    get_shape(),
                    " != dst: ",
                    dst->get_shape(),
                    ")");
    const auto& shape = get_shape();
    auto* src_data = static_cast<const uint8_t*>(data());
    auto* dst_data = static_cast<uint8_t*>(dst->data());
    ov::Strides src_strides{get_byte_size()};
    ov::Strides dst_strides{dst->get_byte_size()};
    ov::Shape cur_pos{0};
    ov::Shape max_pos{1};

    if (get_element_type().bitwidth() < 8 || (get_strides() == dst->get_strides() && is_continuous()) ||
        (is_scalar(get_shape()) && is_scalar(dst->get_shape()))) {
        // OpenVINO doesn't support strides for LP types
        // or both tensors have default strides
        // Strides and positions already initialized
    } else {
        // Tensors have default strides
        const auto& type = get_element_type();
        std::vector<size_t> strides(shape.size());
        if (!shape.empty()) {
            strides[shape.size() - 1] = 1;
        }
        auto size = shape.size();
        for (size_t i = 1; i < size; i++) {
            strides[size - i - 1] = strides[size - i] * shape[size - i];
        }

        ov::Strides default_strides(strides.size());
        for (size_t i = 0; i < strides.size(); ++i)
            default_strides[i] = strides[i] * type.size();

        src_strides = get_strides();
        dst_strides = dst->get_strides();

        ov::Strides src_str, dst_str;

        // Calculate src and dst shapes
        bool found_step = false;
        for (size_t i = 0; i < shape.size(); i++) {
            size_t inverted_idx = shape.size() - i - 1;
            if (!found_step) {
                if (default_strides[inverted_idx] == src_strides[inverted_idx] &&
                    src_strides[inverted_idx] == dst_strides[inverted_idx]) {
                    continue;
                } else {
                    found_step = true;
                    size_t strides_size = inverted_idx + 1;
                    // Set right size
                    src_str.resize(strides_size + 1);
                    dst_str.resize(strides_size + 1);
                    max_pos.resize(strides_size + 1);
                    cur_pos.resize(strides_size + 1);
                    // In case of default continuous strides we can copy several elements
                    // In other case only one element
                    size_t dim = 1;
                    size_t strides = type.size();

                    if (strides_size < default_strides.size()) {
                        strides = default_strides[strides_size];
                        dim = get_shape()[strides_size];
                    }
                    src_str[strides_size] = strides;
                    dst_str[strides_size] = strides;
                    max_pos[strides_size] = dim;
                    cur_pos[strides_size] = 0;
                }
            }
            src_str[inverted_idx] = src_strides[inverted_idx];
            dst_str[inverted_idx] = dst_strides[inverted_idx];
            max_pos[inverted_idx] = shape[inverted_idx];
            cur_pos[inverted_idx] = 0;
        }
        src_strides = src_str;
        dst_strides = dst_str;
    }

    const auto update_index = [](const ov::Shape& pos, const ov::Shape& shape, const ov::Strides& strides) {
        size_t offset = 0;

        for (size_t i = 0; i < pos.size(); i++) {
            offset += pos[i] * strides[i];
        }
        return offset;
    };

    bool finish = false;
    for (size_t dst_idx = 0, src_idx = 0; !finish;) {
        memcpy(dst_data + dst_idx, src_data + src_idx, src_strides[src_strides.size() - 1]);
        // update indexes
        for (size_t i = 0; i < cur_pos.size(); i++) {
            size_t inverted_idx = cur_pos.size() - i - 1;
            cur_pos[inverted_idx]++;
            if (cur_pos[inverted_idx] != max_pos[inverted_idx]) {
                break;
            }
            if (inverted_idx)
                cur_pos[inverted_idx] = 0;
            else
                finish = true;
        }
        src_idx = update_index(cur_pos, max_pos, src_strides);
        dst_idx = update_index(cur_pos, max_pos, dst_strides);
    }
}

}  // namespace ov
