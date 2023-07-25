// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "ctc_greedy_decoder_inst.h"
#include "ctc_greedy_decoder/ctc_greedy_decoder_kernel_selector.h"
#include "ctc_greedy_decoder/ctc_greedy_decoder_kernel_base.h"

namespace cldnn {
namespace ocl {

struct ctc_greedy_decoder_impl : typed_primitive_impl_ocl<ctc_greedy_decoder> {
    using parent = typed_primitive_impl_ocl<ctc_greedy_decoder>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::ctc_greedy_decoder_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::ctc_greedy_decoder_params, kernel_selector::ctc_greedy_decoder_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<ctc_greedy_decoder_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<ctc_greedy_decoder>();
        auto params = get_default_params<kernel_selector::ctc_greedy_decoder_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::ctc_greedy_decoder_optional_params>(impl_param.get_program());

        auto has_second_output = !primitive->second_output.empty();
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
        params.merge_repeated = primitive->ctc_merge_repeated;
        params.blank_index = primitive->blank_index;
        params.outputs_num = has_second_output ? 2 : 1;

        if (params.outputs_num == 2) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        }
        return {params, optional_params};
    }
};

namespace detail {

attach_ctc_greedy_decoder_impl::attach_ctc_greedy_decoder_impl() {
    implementation_map<ctc_greedy_decoder>::add(impl_types::ocl, typed_primitive_impl_ocl<ctc_greedy_decoder>::create<ctc_greedy_decoder_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::ctc_greedy_decoder_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ctc_greedy_decoder)
