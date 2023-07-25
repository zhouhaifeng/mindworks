// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fully_connected_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>
#include <algorithm>
#include "utils.hpp"

#include "matmul_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(fully_connected)

namespace {
bool is_batch_after_spatial(const std::string order) {
    bool spatial_found = false;
    for (auto c : order) {
        switch (c) {
            case 'b':
            case 'n':
                return spatial_found;

            case 'x':
            case 'y':
            case 'z':
            case 'w':
            case 's':
                spatial_found = true;
                break;

            default:
                break;
        }
    }
    return false;
}

format::type get_preferred_format(fully_connected_node const& node, const kernel_impl_params& impl_param) {
    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        return node.get_preferred_output_fmt();
    }

    auto input_layout = impl_param.get_input_layout();

    // for 3d output we have to chose bfyx format
    if (impl_param.typed_desc<fully_connected>()->input_size == 3)
        return format::bfyx;

    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        (is_batch_after_spatial(input_layout.format.order()) ||
         input_layout.format == format::bs_f_bsv16 ||
         input_layout.format == format::bs_fs_fsv8_bsv8))
        return format::yxfb;

    bool no_spatial_padding = true;
    // C++ 11 range loop shouldn't be used here because of incorrect iterator functionality in mutable_array_ref<>
    for (size_t i = 0; i < input_layout.data_padding.lower_size().spatial.size(); ++i) {
        no_spatial_padding &= (input_layout.data_padding.lower_size().spatial[i] == 0);
    }
    for (size_t i = 0; i < input_layout.data_padding.upper_size().spatial.size(); ++i) {
        no_spatial_padding &= (input_layout.data_padding.upper_size().spatial[i] == 0);
    }

    if (input_layout.data_type == data_types::f32 &&
        input_layout.format == format::bfyx &&
        no_spatial_padding &&
        input_layout.batch() != 8)
        return format::bfyx;

    auto input_pitches = input_layout.get_pitches();
    if (input_layout.data_type == data_types::f16 &&
        input_layout.format == format::bfyx &&
        no_spatial_padding &&
        input_pitches.batch[0] % 2 == 0 &&
        input_layout.batch() != 16)
        return format::bfyx;

    // this condition tests whether our input is batch>1 in bfyx format, if yes there will be
    // extra reorder between input and this fc from bfyx to yxfb format (so
    // "is_batch_after_spatial" should return true)
    if (data_type_traits::is_floating_point(input_layout.data_type) &&
        input_layout.format == format::bfyx &&
        input_layout.batch() > 1)
        return format::yxfb;

    return format::bfyx;
}

}  // namespace

layout fully_connected_inst::calc_output_layout(fully_connected_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<fully_connected>();

    auto input_layout = impl_param.get_input_layout();
    auto input_pshape = input_layout.get_partial_shape();
    auto weights_layout = *impl_param.weights_layout;
    auto weights_pshape = weights_layout.get_partial_shape();
    auto output_type = input_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && desc->output_data_types[0])
        output_type = *desc->output_data_types[0];

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    auto reshape_to_2d = [](const ov::PartialShape& shape, int64_t feature) {
        auto staticShape = shape.to_shape();
        size_t total = std::accumulate(staticShape.begin(), staticShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
        return reshapeSize;
    };

    int64_t feature = input_pshape[std::min(desc->input_size, static_cast<size_t>(4)) - 1].get_length();
    if (desc->input_size == 3) {
        feature = std::max({input_layout.spatial(0), input_layout.spatial(1), input_layout.spatial(2)});
    }

    if (desc->input_size > 3) {
       input_layout.set_partial_shape(reshape_to_2d(input_pshape, feature));
    }
    if (weights_pshape.size() != 2) {
        weights_layout.set_partial_shape(reshape_to_2d(weights_pshape, feature));
    }

    auto output_size = tensor(input_layout.batch(), weights_layout.batch(), 1, 1);
    if (desc->input_size == 3) {
        output_size = tensor(input_layout.batch(), input_layout.feature(), 1, weights_layout.batch());
    }
    format output_format = get_preferred_format(node, impl_param);

    return layout(output_type, output_format, output_size);
}

template<typename ShapeType>
std::vector<layout> fully_connected_inst::calc_output_layouts(fully_connected_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<fully_connected>();
    auto input_layout = impl_param.get_input_layout();
    auto weights_layout = *impl_param.weights_layout;

    auto output_type = input_layout.data_type;
    if (data_type_traits::is_i8_u8(output_type) && desc->output_data_types[0])
        output_type = *desc->output_data_types[0];

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ov::op::v0::MatMul op;
    op.set_transpose_b(true);
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        weights_layout.get<ShapeType>()
    };

    std::vector<ShapeType> output_shapes = ov::op::v0::shape_infer(&op, input_shapes);

    bool is_static = input_layout.is_static() && weights_layout.is_static();
    bool allow_new_shape_infer = impl_param.get_program().get_config().get_property(ov::intel_gpu::allow_new_shape_infer);
    format::type output_format = is_static && !allow_new_shape_infer ? get_preferred_format(node, impl_param) :
                                              input_layout.format.value;

    return { layout{output_shapes[0], output_type, output_format} };
}


kernel_impl_params fully_connected_inst::get_fake_aligned_params(kernel_impl_params const& orig_impl_param) {
    // fc_tiled_opt kernel is optimized for row shape aligned by 8.
    // Thus, use fake aligned shape at kernel execution for better performance.
    auto orig_input_layout = orig_impl_param.get_input_layout();
    auto orig_output_layout = orig_impl_param.get_output_layout();
    OPENVINO_ASSERT(orig_input_layout.is_static() && orig_output_layout.is_static(),
                    "in/out layouts should be static for fake alignment!");
    if (orig_input_layout.format == format::bfyx && orig_output_layout.format == format::bfyx) {
        auto updated_param = orig_impl_param;
        auto input_shape = orig_input_layout.get_partial_shape().to_shape();
        auto input_row_idx = input_shape.size() - 2;
        auto output_shape = orig_output_layout.get_partial_shape().to_shape();
        auto output_row_idx = output_shape.size() - 2;

        // Vector by matrix multiplication sometimes works slower if we align it
        if (input_shape[input_row_idx] == 1 && output_shape[output_row_idx] == 1 && input_shape[input_shape.size() - 1] >= 1024) {
            return std::move(orig_impl_param);
        }

        input_shape[input_row_idx] = align_to(input_shape[input_row_idx], 8);
        output_shape[output_row_idx] = align_to(output_shape[output_row_idx], 8);

        updated_param.input_layouts[0] = layout(ov::PartialShape(input_shape),
                                                orig_input_layout.data_type,
                                                orig_input_layout.format,
                                                orig_input_layout.data_padding);
        updated_param.output_layouts[0] = layout(ov::PartialShape(output_shape),
                                             orig_output_layout.data_type,
                                             orig_output_layout.format,
                                             orig_output_layout.data_padding);
        return updated_param;
    }
    return std::move(orig_impl_param);
}

template std::vector<layout> fully_connected_inst::calc_output_layouts<ov::PartialShape>(fully_connected_node const& node,
                                                                                         const kernel_impl_params& impl_param);

std::string fully_connected_inst::to_string(fully_connected_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto weights_id = desc->weights;

    std::stringstream primitive_description;

    json_composite fc_info;
    fc_info.add("weights id", weights_id);
    fc_info.add("bias id", bias_id);

    node_info->add("fully connected info", fc_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fully_connected_inst::typed_primitive_inst(network& network, fully_connected_node const& node)
    : parent(network, node) { }
}  // namespace cldnn
