// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random_uniform_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>
#include <data_inst.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(random_uniform)

random_uniform_inst::typed_primitive_inst(network& network, random_uniform_node const &node)
: parent(network, node) {
}

layout random_uniform_inst::calc_output_layout(random_uniform_node const &node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<random_uniform>();
    return {*primitive->output_data_types[0], primitive->output_format, primitive->output_shape};
}

std::string random_uniform_inst::to_string(random_uniform_node const &node) {
    auto node_info = node.desc_to_json();
    json_composite random_uniform_info;
    random_uniform_info.add("input id", node.input().id());
    random_uniform_info.add("min_value id", node.input(1).id());
    random_uniform_info.add("max_value  id", node.input(2).id());
    random_uniform_info.add("global_seed", node.get_primitive()->global_seed);
    random_uniform_info.add("op_seed", node.get_primitive()->op_seed);
    node_info->add("random uniform info", random_uniform_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
