// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "snippets/target_machine.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace intel_cpu {

class CPUTargetMachine : public snippets::TargetMachine {
public:
    CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa);

    bool is_supported() const override;
    snippets::code get_snippet() const override;
    size_t get_lanes() const override;

private:
    std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h;
    dnnl::impl::cpu::x64::cpu_isa_t isa;
};

class CPUGenerator : public snippets::Generator {
public:
    CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa);

protected:
    opRegType get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const override;
};

}   // namespace intel_cpu
}   // namespace ov
