// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_inst.h"
#include "pad_shape_inference.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(border)

layout border_inst::calc_output_layout(border_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for border_node!");
    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;
    auto desc = impl_param.typed_desc<border>();

    auto dims_format = format::adjust_to_rank(format::bfyx, input_layout.get_rank());
    auto new_dims = input_layout.get_dims();

    for (size_t i = 0; i < new_dims.size(); ++i) {
        new_dims[i] += (i < desc->pads_begin.size()) ? desc->pads_begin[i] : 0;
        new_dims[i] += (i < desc->pads_end.size()) ? desc->pads_end[i] : 0;
    }
    return layout{ input_layout.data_type, input_format, tensor(dims_format, new_dims) };
}

template<typename ShapeType>
std::vector<layout> border_inst::calc_output_layouts(border_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<border>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto output_type = input0_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ov::op::v1::Pad op;
    op.set_pad_mode(desc->pad_mode);

    const bool is_begin_mem = (desc->non_constant_input_mask & border::PAD_NON_CONST_INPUT::BEGIN);
    const bool is_end_mem = (desc->non_constant_input_mask & border::PAD_NON_CONST_INPUT::END);

    layout pads_begin_layout, pads_end_layout;
    if (is_begin_mem) {
        pads_begin_layout = impl_param.get_input_layout(1);
    }
    if (is_end_mem) {
        pads_end_layout = is_begin_mem ? impl_param.get_input_layout(2) : impl_param.get_input_layout(1);
    }

    ShapeType pads_begin_shape = is_begin_mem ? pads_begin_layout.get<ShapeType>() : ov::Shape{ desc->pads_begin.size() };
    ShapeType pads_end_shape = is_end_mem ? pads_end_layout.get<ShapeType>() : ov::Shape{ desc->pads_end.size() };
    std::vector<ShapeType> output_shapes;
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        pads_begin_shape,
        pads_end_shape,
    };

    auto& memory_deps = impl_param.memory_deps;
    std::map<size_t, ngraph::HostTensorPtr> const_data;
    auto ta = ov::make_tensor_accessor(const_data);

    if ((is_begin_mem && memory_deps.count(1)) && (is_end_mem && memory_deps.count(2))) {
        auto pads_begin_mem = memory_deps.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> pads_begin_lock(pads_begin_mem, impl_param.get_stream());
        const_data.emplace(1, make_host_tensor(pads_begin_mem->get_layout(), pads_begin_lock.data()));

        auto pads_end_mem = memory_deps.at(2);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> pads_end_lock(pads_end_mem, impl_param.get_stream());
        const_data.emplace(2, make_host_tensor(pads_end_mem->get_layout(), pads_end_lock.data()));

        output_shapes = ov::op::shape_infer(&op, input_shapes, ta);
    } else if ((is_begin_mem || is_end_mem) && memory_deps.count(1)) {
        if (is_begin_mem) {
            auto pads_begin_mem = memory_deps.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> pads_begin_lock(pads_begin_mem, impl_param.get_stream());
            const_data.emplace(1, make_host_tensor(pads_begin_mem->get_layout(), pads_begin_lock.data()));

            auto pads_end_data = desc->pads_end;
            auto pads_end_tensor = make_host_tensor({pads_end_shape, data_types::i64, format::bfyx}, static_cast<void*>(pads_end_data.data()));
            const_data.emplace(2, pads_end_tensor);

            output_shapes = ov::op::shape_infer(&op, input_shapes, ta);
        } else {
            auto pads_begin_data = desc->pads_begin;
            auto pads_begin_tensor = make_host_tensor({pads_begin_shape, data_types::i64, format::bfyx}, static_cast<void*>(pads_begin_data.data()));
            const_data.emplace(1, pads_begin_tensor);

            auto pads_end_mem = memory_deps.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> pads_end_lock(pads_end_mem, impl_param.get_stream());
            const_data.emplace(2, make_host_tensor(pads_end_mem->get_layout(), pads_end_lock.data()));

            output_shapes = ov::op::shape_infer(&op, input_shapes, ta);
        }
    } else {
        std::ptrdiff_t val = desc->pad_value;

        auto pads_begin_data = desc->pads_begin;
        if (is_begin_mem && desc->pad_mode == ov::op::PadMode::CONSTANT) {
            pads_begin_data = {val, val, val, val};
        }
        auto pads_begin_tensor = make_host_tensor({pads_begin_shape, data_types::i64, format::bfyx}, static_cast<void*>(pads_begin_data.data()));
        const_data.emplace(1, pads_begin_tensor);

        auto pads_end_data = desc->pads_end;
        if (is_end_mem && desc->pad_mode == ov::op::PadMode::CONSTANT) {
            pads_end_data = {val, val, val, val};
        }
        auto pads_end_tensor = make_host_tensor({pads_end_shape, data_types::i64, format::bfyx}, static_cast<void*>(pads_end_data.data()));
        const_data.emplace(2, pads_end_tensor);

        output_shapes = ov::op::shape_infer(&op, input_shapes, ta);
    }

    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> border_inst::calc_output_layouts<ov::PartialShape>(border_node const& node, const kernel_impl_params& impl_param);

std::string border_inst::to_string(border_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite border_info;
    border_info.add("pads_begin", desc->pads_begin);
    border_info.add("pads_end", desc->pads_end);
    border_info.add("pad mode", desc->pad_mode);
    border_info.add("pad value", std::to_string(desc->pad_value));

    node_info->add("border info", border_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

border_inst::typed_primitive_inst(network& network, border_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();
    if (input_layout.is_dynamic()) {
        return;
    }

    const auto& input_sizes = input_layout.get_dims();
    auto pad_mode = argument->pad_mode;

    // Check if sizes of border are in proper range.
    CLDNN_ERROR_BOOL(node.id(),
                     "pads_begin border sizes",
                     std::any_of(argument->pads_begin.begin(), argument->pads_begin.end(),
                                 [](std::ptrdiff_t pad) {
                                    return pad < 0;
                                }),
                     "Invalid border size: negative value");
    CLDNN_ERROR_BOOL(node.id(),
                     "pads_end border sizes",
                     std::any_of(argument->pads_end.begin(), argument->pads_end.end(),
                                 [](std::ptrdiff_t pad) {
                                    return pad < 0;
                                }),
                     "Invalid border size: negative value");

    if (pad_mode == ov::op::PadMode::SYMMETRIC) {
        bool valid_pads = true;

        for (size_t i = 0; i < argument->pads_begin.size(); ++i) {
            valid_pads &= argument->pads_begin[i] <= input_sizes[i];
            valid_pads &= argument->pads_end[i] <= input_sizes[i];
        }
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_begin/pads_end border sizes",
                         !valid_pads,
                         "Not enough data in input to create SYMMETRIC border of specified size");
    } else if (pad_mode == ov::op::PadMode::REFLECT) {
        bool valid_pads = true;

        for (size_t i = 0; i < argument->pads_begin.size(); ++i) {
            valid_pads &= argument->pads_begin[i] < input_sizes[i];
            valid_pads &= argument->pads_end[i] < input_sizes[i];
        }
        CLDNN_ERROR_BOOL(node.id(),
                         "pads_begin/pads_end border sizes",
                         !valid_pads,
                         "Not enough data in input to create REFLECT border of specified size");
    }
}
}  // namespace cldnn
