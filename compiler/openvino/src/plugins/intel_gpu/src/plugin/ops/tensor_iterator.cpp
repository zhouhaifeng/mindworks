// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/plugin.hpp"

#include <cpp/ie_cnn_network.h>

#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"

#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/graph/topology.hpp"

#include <vector>
#include <algorithm>

using TensorIterator = ngraph::op::v0::TensorIterator;

namespace ov {
namespace intel_gpu {

template<class DATA_TYPE>
static DATA_TYPE CreateScalarData(Program &p, const cldnn::primitive_id& id, int64_t num) {
    auto mem = p.get_engine().allocate_memory({ cldnn::data_types::i64, cldnn::format::bfyx, { 1, 1, 1, 1 } });
    cldnn::mem_lock<int64_t> ptr{mem, p.get_engine().get_service_stream()};
    *ptr.begin() = num;
    return {id, mem};
}

static cldnn::mutable_data CreateAdditionalOutputData(Program &p, const std::shared_ptr<ngraph::Node>& op,
                                            const cldnn::primitive_id& id, const cldnn::primitive_id& input,
                                            const int32_t output_idx) {
    const auto precision = cldnn::element_type_to_data_type(op->get_output_element_type(output_idx));
    const auto format = cldnn::format::get_default_format(op->get_output_shape(output_idx).size());
    const auto tensor = tensor_from_dims(op->get_output_shape(output_idx));
    cldnn::layout output_layout = cldnn::layout(precision, format, tensor);
    auto mem = p.get_engine().allocate_memory(output_layout);
    auto md = cldnn::mutable_data(id, {cldnn::input_info(input)}, mem); // cldnn::data cannot set dependency
    return md;
}

static void CreateTensorIteratorOp(Program &p, const std::shared_ptr<TensorIterator> &op) {
    auto inputs = p.GetInputInfo(op);

    // get body topology from ngraph function
    InferenceEngine::CNNNetwork body_network(op->get_body());
    Program body_program(body_network, p.get_engine(), p.get_config(), true);
    auto body_topology = *body_program.GetTopology();

    // setup input_primitive_maps/ output_primitive_maps and back_edges
    const auto& loop_input_descs = op->get_input_descriptions();
    const auto& loop_output_descs = op->get_output_descriptions();
    const auto& body_inputs = op->get_body()->get_parameters();
    const auto& body_outputs = op->get_body()->get_results();

    std::vector<cldnn::loop::io_primitive_map> input_primitive_maps;
    std::vector<cldnn::loop::io_primitive_map> output_primitive_maps;
    std::vector<cldnn::loop::backedge_mapping> back_edges;
    std::map<cldnn::primitive_id, cldnn::primitive_id> reordered_output_ids;

    // set input mapping & back edges
    for (const auto& loop_input_desc : loop_input_descs) {
        const cldnn::primitive_id& external_id = inputs.at(loop_input_desc->m_input_index).pid;
        auto& body_input = body_inputs.at(loop_input_desc->m_body_parameter_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_input);

        // set input mapping
        if (const auto& sliceInfo =
            std::dynamic_pointer_cast<TensorIterator::SliceInputDescription>(loop_input_desc)) {
            // sliced input
            input_primitive_maps.emplace_back(external_id, internal_id, sliceInfo->m_axis,
                sliceInfo->m_start, sliceInfo->m_end, sliceInfo->m_stride);
        } else {
            // input without slicing
            input_primitive_maps.emplace_back(external_id, internal_id);
        }

        // set back edges
        if (const auto& mergedInput =
            std::dynamic_pointer_cast<TensorIterator::MergedInputDescription>(loop_input_desc)) {
            // backedge
            const auto& to = body_inputs.at(mergedInput->m_body_parameter_index);
            const auto& from = body_outputs.at(mergedInput->m_body_value_index);

            cldnn::primitive_id to_id = layer_type_name_ID(to);
            cldnn::primitive_id from_id = layer_type_name_ID(from);

            // reset output data type because the data types of the outputs of the
            // body topology are always FP32 regardless of ngraph data type
            {
                const auto from_prim = body_topology.at(from_id);
                const auto& to_ngraph_type = to->get_element_type();
                const auto to_cldnn_type = cldnn::element_type_to_data_type(to_ngraph_type);
                from_prim->output_data_types = {to_cldnn_type};
            }
            back_edges.emplace_back(from_id, to_id);
        }
    }

    // set trip count, initial execution condition, num iteration primitives
    // they should be mutable_data to prevent from being optimized out
    std::string layerName = layer_type_name_ID(op);
    const cldnn::primitive_id trip_count_id = layerName + "_tripCount";
    const int64_t num_iterations = op->get_num_iterations();
    if (num_iterations < 0) {
        throw std::runtime_error("tensor iterator's num_iteration cannot be negative");
    }
    {
        cldnn::data trip_count = CreateScalarData<cldnn::data>(p, trip_count_id, num_iterations);
        p.add_primitive(*op, trip_count);
    }
    const cldnn::primitive_id execution_condition_id = layerName + "_initialExecutionCondition";
    {
        cldnn::mutable_data execution_condition = CreateScalarData<cldnn::mutable_data>(p, execution_condition_id, 1);
        p.add_primitive(*op, execution_condition);
    }
    const cldnn::primitive_id num_iteration_id = layerName + "_numIteration";
    {
        cldnn::mutable_data num_iteration = CreateScalarData<cldnn::mutable_data>(p, num_iteration_id, 0);
        p.add_primitive(*op, num_iteration);
    }

    // set output mapping
    for (const auto& loop_output_desc : loop_output_descs) {
        const uint64_t output_idx = loop_output_desc->m_output_index;

        // Add additional mutable_data for multiple outputs
        // primitive ID should be <TI primitive ID>.<output_idx> if output_idx > 0
        // otherwise primitive ID should be equals to TI primitive ID
        const std::string layerNameWithIndex = layerName + ".out" + std::to_string(output_idx);
        std::string external_id;
        if (output_idx > 0) {
            cldnn::mutable_data output_data = CreateAdditionalOutputData(p, op, layerNameWithIndex, layerName, output_idx);
            p.add_primitive(*op, output_data);
            external_id = layerNameWithIndex;
        } else {
            p.primitive_ids[layerNameWithIndex] = layerName;
            p.primitive_ids[layerName] = layerName;
            external_id = layerName;
        }
        const auto& body_output = body_outputs.at(loop_output_desc->m_body_value_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_output);

        // update primitive_map
        if (const auto& concatOutput =
            std::dynamic_pointer_cast<TensorIterator::ConcatOutputDescription>(loop_output_desc)) {
            // output which requires concatenation
            output_primitive_maps.emplace_back(external_id, internal_id, concatOutput->m_axis,
                concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
        }
        if (std::dynamic_pointer_cast<TensorIterator::BodyOutputDescription>(loop_output_desc)) {
            // output which requires no concatenation
            output_primitive_maps.emplace_back(external_id, internal_id);
        }
    }

    const cldnn::loop loopPrimitive(
        layerName,              /* layer name of this primitive (output id) */
        inputs,        /* inputs of this layer */
        body_topology,          /* body network */
        trip_count_id,          /* trip_count data in outer network, always same as num_iterations in TI */
        execution_condition_id, /* initial_execution_condition data in outer network, always true in TI */
        num_iteration_id,       /* actual number of iteration data in body network */
        input_primitive_maps,         /* input mappings connecting outer network and inner network */
        output_primitive_maps,        /* output mappings connecting outer network and inner network */
        back_edges,             /* back edge mapping */
        num_iterations,         /* max iteration, i.e. length of iteration axis */
        "",
        "");

    p.add_primitive(*op, loopPrimitive);
}

REGISTER_FACTORY_IMPL(v0, TensorIterator);

}  // namespace intel_gpu
}  // namespace ov
