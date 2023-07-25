// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lstm.hpp"
#include "primitive_inst.h"

#include <string>
#include <vector>

namespace cldnn {
template <>
struct typed_program_node<lstm> : public typed_program_node_base<lstm> {
    using parent = typed_program_node_base<lstm>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& weights() const { return get_dependency(1); }
    program_node& recurrent() const { return get_dependency(2); }
    program_node& bias() const { return get_dependency(3); }
    program_node& inital_hidden() const { return get_dependency(bias_term() ? 4 : 3); }
    program_node& inital_cell() const {
        // This doesn't scale. We should use a map to get the dependencies index at primitive level
        return get_dependency(bias_term() ? (initial_hidden_term() ? 5 : 4) : (initial_hidden_term() ? 4 : 2));
    }
    program_node& peepholes() const { return get_dependency(6); }
    bool bias_term() const { return !get_primitive()->bias.empty(); }
    bool peepholes_term() const { return !get_primitive()->peepholes.empty(); }
    bool initial_hidden_term() const { return !get_primitive()->initial_hidden.empty(); }
    bool initial_cell_term() const { return !get_primitive()->initial_cell.empty(); }
    std::vector<activation_func> activations() const { return get_primitive()->activations; }
    std::vector<activation_additional_params> activation_params() const {
        return get_primitive()->activation_params;
    }
    size_t sequence_len() const { return get_primitive()->input.size(); }
};

using lstm_node = typed_program_node<lstm>;

template <>
class typed_primitive_inst<lstm> : public typed_primitive_inst_base<lstm> {
    using parent = typed_primitive_inst_base<lstm>;
    using parent::parent;

public:
    static layout calc_output_layout(lstm_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_node const& node);

public:
    typed_primitive_inst(network& network, lstm_node const& node);

    memory& weights_memory() const { return dep_memory(1); }
    memory& recurrent_memory() const { return dep_memory(2); }
    memory& bias_memory() const { return dep_memory(3); }
    memory& initial_hidden_memory() const { return dep_memory(bias_term() ? 4 : 3); }
    memory& initial_cell_memory() const {
        return dep_memory(bias_term() ? (initial_hidden_term() ? 5 : 4) : (initial_hidden_term() ? 4 : 2));
    }
    memory& peepholes_memory() const { return dep_memory(6); }
    bool bias_term() const { return !argument->bias.empty(); }
    bool peepholes_term() const { return !argument->peepholes.empty(); }
    bool initial_hidden_term() const { return !argument->initial_hidden.empty(); }
    bool initial_cell_term() const { return !argument->initial_cell.empty(); }
    std::vector<activation_func> activations() const { return argument->activations; }
    std::vector<activation_additional_params> activation_params() const { return argument->activation_params; }
};

using lstm_inst = typed_primitive_inst<lstm>;

}  // namespace cldnn
