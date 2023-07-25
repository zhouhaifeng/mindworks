// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "concatenation_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "openvino/op/concat.hpp"

namespace cldnn {
namespace cpu {

struct concatenation_impl : public typed_primitive_impl<concatenation> {
    using parent = typed_primitive_impl<concatenation>;
    using parent::parent;

    int64_t axis;

    std::shared_ptr<ov::op::v0::Concat> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<concatenation_impl>(*this);
    }

    concatenation_impl() : parent("concatenation_cpu_impl") {}

    explicit concatenation_impl(const concatenation_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<concatenation>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<concatenation>();
        axis = node.get_primitive()->axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> axis;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, concatenation_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "concat::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);

        auto params = instance.get_impl_params();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        for (auto input_layout : instance.get_impl_params()->input_layouts)
            OPENVINO_ASSERT(input_layout.data_type == instance.get_impl_params()->get_output_layout().data_type,
                            "[GPU] Couldn't create concat operation: unsupported mixed inputs/output data types");

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++) {
            auto& dep = instance.dependencies().at(i);
            if (dep.first->get_output_layout().count() > 0)
                input_mem_ptrs.push_back(instance.dep_memory_ptr(i));
        }

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::read> output_lock(output_mem_ptr, stream);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));

        if (!op) {
            op = std::make_shared<ov::op::v0::Concat>();
            op->set_axis(instance.get_typed_desc<concatenation>()->axis);
        }

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute concat primitive with id ", instance.id());

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const concatenation_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<concatenation_impl>();
    }
};


namespace detail {

attach_concatenation_impl::attach_concatenation_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<concatenation>::add(impl_types::cpu, shape_types::static_shape, concatenation_impl::create, types, formats);
    implementation_map<concatenation>::add(impl_types::cpu, shape_types::dynamic_shape, concatenation_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::concatenation_impl)
