// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "compiled_model.hpp"
#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "plugin.hpp"

ov::hetero::InferRequest::InferRequest(const std::shared_ptr<const ov::hetero::CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    for (auto&& comp_model_desc : compiled_model->m_compiled_submodels) {
        auto& comp_model = comp_model_desc.compiled_model;
        m_subrequests.push_back({comp_model->create_infer_request(), comp_model._so});
    }

    for (size_t i = 0; i < compiled_model->inputs().size(); i++) {
        const auto& port = compiled_model->inputs()[i];
        const auto& submodel_idx = compiled_model->m_inputs_to_submodels_inputs[i].first;
        m_port_to_subrequest_idx[port] = submodel_idx;
    }
    for (size_t i = 0; i < compiled_model->outputs().size(); i++) {
        const auto& port = compiled_model->outputs()[i];
        const auto& submodel_idx = compiled_model->m_outputs_to_submodels_outputs[i].first;
        m_port_to_subrequest_idx[port] = submodel_idx;
    }

    for (const auto& kvp : compiled_model->m_submodels_input_to_prev_output) {
        const auto& submodel_idx_in = kvp.first.first;
        const auto& port_idx_in = kvp.first.second;
        const auto& submodel_idx_out = kvp.second.first;
        const auto& port_idx_out = kvp.second.second;

        const auto& output_port = m_subrequests[submodel_idx_out]->get_compiled_model()->outputs()[port_idx_out];
        const auto& output_tensor = m_subrequests[submodel_idx_out]->get_tensor(output_port);
        const auto& input_port = m_subrequests[submodel_idx_in]->get_compiled_model()->inputs()[port_idx_in];
        m_subrequests[submodel_idx_in]->set_tensor(input_port, output_tensor);
    }
}

ov::hetero::InferRequest::~InferRequest() = default;

ov::SoPtr<ov::IAsyncInferRequest> ov::hetero::InferRequest::get_request(const ov::Output<const ov::Node>& port) const {
    auto check_nodes = [](const ov::Node* node1, const ov::Node* node2) {
        return node1 == node2 ||
               (node1->get_friendly_name() == node2->get_friendly_name() &&
                node1->get_type_info() == node2->get_type_info() &&
                node1->outputs().size() == node2->outputs().size() && node1->inputs().size() == node2->inputs().size());
    };

    for (const auto& kvp : m_port_to_subrequest_idx) {
        if (kvp.first.get_index() == port.get_index() && kvp.first.get_names() == port.get_names() &&
            check_nodes(kvp.first.get_node(), port.get_node())) {
            return m_subrequests[kvp.second];
        }
    }
    OPENVINO_THROW("Cannot find infer request for port ", port);
}

ov::SoPtr<ov::ITensor> ov::hetero::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    return get_request(port)->get_tensor(port);
}

void ov::hetero::InferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                          const ov::SoPtr<ov::ITensor>& tensor) {
    get_request(port)->set_tensor(port, tensor);
}

std::vector<ov::SoPtr<ov::ITensor>> ov::hetero::InferRequest::get_tensors(
    const ov::Output<const ov::Node>& port) const {
    return get_request(port)->get_tensors(port);
}

void ov::hetero::InferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                           const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return get_request(port)->set_tensors(port, tensors);
}

void ov::hetero::InferRequest::check_tensors() const {
    // Ignore `check_tensor` of inputs and outputs of Hetero Compiled Model because
    // `m_tensors` are not allocated
    return;
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::hetero::InferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> variable_states = {};
    for (const auto& request : m_subrequests) {
        OPENVINO_ASSERT(request);
        for (auto&& state : request->query_state()) {
            if (!state._so)
                state._so = request._so;
            variable_states.emplace_back(state);
        }
    }
    return variable_states;
}

void ov::hetero::InferRequest::infer() {
    for (auto&& request : m_subrequests) {
        OPENVINO_ASSERT(request);
        request->infer();
    }
}

std::vector<ov::ProfilingInfo> ov::hetero::InferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    for (size_t i = 0; i < m_subrequests.size(); ++i) {
        auto&& subreq_info = m_subrequests[i]->get_profiling_info();
        for (auto&& rec : subreq_info)
            rec.node_name = std::string("subgraph") + std::to_string(i) + ": " + rec.node_name;
        info.insert(info.end(), subreq_info.begin(), subreq_info.end());
    }
    return info;
}
