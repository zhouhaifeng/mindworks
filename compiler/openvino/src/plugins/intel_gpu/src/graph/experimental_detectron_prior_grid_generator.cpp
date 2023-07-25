// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental_detectron_prior_grid_generator_inst.h>
#include <primitive_type_base.h>

#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(experimental_detectron_prior_grid_generator)

layout experimental_detectron_prior_grid_generator_inst::calc_output_layout(
    const experimental_detectron_prior_grid_generator_node& node, kernel_impl_params const& impl_param) {
    const layout data_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<experimental_detectron_prior_grid_generator>();
    if (desc->flatten) {
        return layout(data_layout.data_type,
                      format::bfyx,
                      {static_cast<int>(desc->featmap_width * desc->featmap_height * data_layout.batch()), 4, 1, 1});
    } else {
        return layout(data_layout.data_type,
                      format::bfyx,
                      {static_cast<int>(desc->featmap_height),
                       static_cast<int>(desc->featmap_width),
                       4,
                       static_cast<int>(data_layout.batch())});
    }
}

std::string experimental_detectron_prior_grid_generator_inst::to_string(
    experimental_detectron_prior_grid_generator_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite experimental_detectron_prior_grid_generator_info;
    experimental_detectron_prior_grid_generator_info.add("flatten", node.get_primitive()->flatten);
    experimental_detectron_prior_grid_generator_info.add("h", node.get_primitive()->h);
    experimental_detectron_prior_grid_generator_info.add("w", node.get_primitive()->w);
    experimental_detectron_prior_grid_generator_info.add("stride_x", node.get_primitive()->stride_x);
    experimental_detectron_prior_grid_generator_info.add("stride_y", node.get_primitive()->stride_y);
    node_info->add("experimental_detectron_prior_grid_generator_info",
                   experimental_detectron_prior_grid_generator_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
