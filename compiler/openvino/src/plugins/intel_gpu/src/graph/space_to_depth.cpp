// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(space_to_depth)

layout space_to_depth_inst::calc_output_layout(space_to_depth_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<space_to_depth>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    const size_t block_size = desc->block_size;
    auto depth_mode = desc->mode;

    auto output_type = input_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    if (depth_mode != space_to_depth::depth_first && depth_mode != space_to_depth::blocks_first)
        CLDNN_ERROR_MESSAGE(desc->id,
                            "Invalid mode for spaceToDepth: must be \"blocks_first\" or \"depth_first\" only");

    if (block_size == 0)
        CLDNN_ERROR_MESSAGE(desc->id,
                            "Invalid spaceToDepth block_size value (should be >= 1). Actual block size is" +
                                std::to_string(block_size));

    if (input_layout.spatial(0) % block_size != 0 || input_layout.spatial(1) % block_size != 0)
        CLDNN_ERROR_MESSAGE(
            desc->id,
            "Sizes of spatials x, y must be divisible by block size. Actual spatial sizes are " +
                std::to_string(input_layout.spatial(0)) + ", " + std::to_string(input_layout.spatial(1)) +
                    " (x, y). Actual block size is " + std::to_string(block_size));


    if (input_layout.format.dimension() == 5) {
        if (input_layout.spatial(2) % block_size != 0)
        CLDNN_ERROR_MESSAGE(
            desc->id,
            "Sizes of spatials z must be divisible by block size. Actual spatial sizes are " +
                std::to_string(input_layout.spatial(2)) +
                    " (z). Block size is " + std::to_string(block_size));

        const size_t feature = input_layout.feature() * block_size * block_size * block_size;
        const size_t z = input_layout.spatial(2) / block_size;
        const size_t y = input_layout.spatial(1) / block_size;
        const size_t x = input_layout.spatial(0) / block_size;

        return layout{
            output_type,
            input_format,
            tensor(TensorValue(input_layout.batch()), TensorValue(feature), TensorValue(x), TensorValue(y), TensorValue(z))};
    } else {
        const size_t feature = input_layout.feature() * block_size * block_size;
        const size_t y = input_layout.spatial(1) / block_size;
        const size_t x = input_layout.spatial(0) / block_size;

        return layout{
            output_type,
            input_format,
            tensor(TensorValue(input_layout.batch()), TensorValue(feature), TensorValue(x), TensorValue(y))};
    }
}

std::string space_to_depth_inst::to_string(space_to_depth_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    std::string depth_mode = (desc->mode == cldnn::space_to_depth::depth_mode::blocks_first) ?
                             "blocks_first" :
                             "depth_first";

    json_composite space_to_depth_info;
    space_to_depth_info.add("input id", input.id());
    space_to_depth_info.add("mode", depth_mode);
    space_to_depth_info.add("block size", desc->block_size);

    node_info->add("space_to_depth info", space_to_depth_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

space_to_depth_inst::typed_primitive_inst(network& network, space_to_depth_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
