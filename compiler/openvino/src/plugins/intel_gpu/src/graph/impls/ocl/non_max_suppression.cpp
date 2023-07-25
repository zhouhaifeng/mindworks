// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "non_max_suppression_inst.h"
#include "data_inst.h"
#include "non_max_suppression/non_max_suppression_kernel_ref.h"
#include "non_max_suppression/non_max_suppression_kernel_selector.h"

namespace cldnn {
namespace ocl {
struct non_max_suppression_impl : typed_primitive_impl_ocl<non_max_suppression> {
    using parent = typed_primitive_impl_ocl<non_max_suppression>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::non_max_suppression_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::non_max_suppression_params, kernel_selector::non_max_suppression_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<non_max_suppression_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<non_max_suppression>& instance) const override {
        kernel_arguments_data args;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_num_select_per_class() && !instance.num_select_per_class_inst()->is_constant()) {
            args.inputs.push_back(instance.num_select_per_class_mem());
        }

        if (instance.has_iou_threshold() && !instance.iou_threshold_inst()->is_constant()) {
            args.inputs.push_back(instance.iou_threshold_mem());
        }

        if (instance.has_score_threshold() && !instance.score_threshold_inst()->is_constant()) {
            args.inputs.push_back(instance.score_threshold_mem());
        }

        if (instance.has_soft_nms_sigma() && !instance.soft_nms_sigma_inst()->is_constant()) {
            args.inputs.push_back(instance.soft_nms_sigma_mem());
        }

        // New API for mutiple outputs support
        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }
        // Legacy APIs using mutable inputs for multiple outputs
        if (instance.has_second_output())
            args.inputs.push_back(instance.second_output_mem());
        if (instance.has_third_output())
            args.inputs.push_back(instance.third_output_mem());

        return args;
    }

public:
    static std::unique_ptr<primitive_impl> create(const non_max_suppression_node& arg, const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<non_max_suppression>();
        auto params = get_default_params<kernel_selector::non_max_suppression_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::non_max_suppression_optional_params>(impl_param.get_program());

        const auto input_scores_idx = 1;
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[input_scores_idx]));

        if (arg.has_num_select_per_class()) {
            cldnn::program_node& node = arg.num_select_per_class_node();
            if (node.is_type<data>()) {
                params.num_select_per_class_type = kernel_selector::base_params::ArgType::Constant;
                params.num_select_per_class = get_value<int>(node);
            } else {
                params.num_select_per_class_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        if (arg.has_iou_threshold()) {
            cldnn::program_node& node = arg.iou_threshold_node();
            if (node.is_type<data>()) {
                params.iou_threshold_type = kernel_selector::base_params::ArgType::Constant;
                params.iou_threshold = get_value<float>(node);
            } else {
                params.iou_threshold_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        if (arg.has_score_threshold()) {
            cldnn::program_node& node = arg.score_threshold_node();
            if (node.is_type<data>()) {
                params.score_threshold_type = kernel_selector::base_params::ArgType::Constant;
                params.score_threshold = get_value<float>(node);
            } else {
                params.score_threshold_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        if (arg.has_soft_nms_sigma()) {
            cldnn::program_node& node = arg.soft_nms_sigma_node();
            if (node.is_type<data>()) {
                params.soft_nms_sigma_type = kernel_selector::base_params::ArgType::Constant;
                params.soft_nms_sigma = get_value<float>(node);
            } else {
                params.soft_nms_sigma_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        auto get_additional_output_node_idx = [&] (bool is_third) {
            size_t offset = 2;
            offset += arg.has_num_select_per_class();
            offset += arg.has_iou_threshold();
            offset += arg.has_score_threshold();
            offset += arg.has_soft_nms_sigma();
            if (is_third)
                offset += arg.has_second_output();
            return offset;
        };

        if (arg.has_second_output()) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[get_additional_output_node_idx(false)]));
            params.has_second_output = true;
        }

        if (arg.has_third_output()) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[get_additional_output_node_idx(true)]));
            params.has_third_output = true;
        }

        if (arg.use_multiple_outputs()) {
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[2]));
            params.use_multiple_outputs = true;
        }

        params.sort_result_descending = primitive->sort_result_descending;
        params.box_encoding = primitive->center_point_box ? kernel_selector::BoxEncodingType::BOX_ENCODING_CENTER
                                                          : kernel_selector::BoxEncodingType::BOX_ENCODING_CORNER;
        if (impl_param.get_program().get_node(primitive->id).is_dynamic()) {
            params.reuse_internal_buffer = true;
        }

        params.set_dynamic_shape_offsets();
        auto& kernel_selector = kernel_selector::non_max_suppression_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(params, optional_params);

        return make_unique<non_max_suppression_impl>(best_kernel);
    }

private:
    template <class T>
    static T get_value(cldnn::program_node& node) {
        T retValue;
        auto mem = node.as<data>().get_attached_memory_ptr();
        auto& stream = node.get_program().get_stream();
        switch (mem->get_layout().data_type) {
        case data_types::f16: {
            mem_lock<half_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<half_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::f32: {
            mem_lock<float, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<float*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i32: {
            mem_lock<int32_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<int32_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i64: {
            mem_lock<int64_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<int64_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        default:
            throw std::runtime_error("Not supported data type.");
        }

        return retValue;
    }
};

namespace detail {

attach_non_max_suppression_impl::attach_non_max_suppression_impl() {
    implementation_map<non_max_suppression>::add(impl_types::ocl,
                                                 non_max_suppression_impl::create,
                                                 {
                                                     std::make_tuple(data_types::i32, format::bfyx),

                                                     std::make_tuple(data_types::f16, format::bfyx),
                                                     std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
                                                     std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
                                                     std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
                                                     std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
                                                     std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),

                                                     std::make_tuple(data_types::f32, format::bfyx),
                                                     std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
                                                     std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
                                                     std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
                                                     std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
                                                     std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
                                                 });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::non_max_suppression_impl)
