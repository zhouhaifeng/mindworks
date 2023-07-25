// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/remarks.hpp"

#include "snippets/op/subgraph.hpp"
#include "snippets/op/convert_saturation.hpp"

#include "snippets/pass/insert_movebroadcast.hpp"
#include "snippets/pass/broadcast_to_movebroadcast.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/pass/convert_constants.hpp"
#include "snippets/pass/convert_power_to_powerstatic.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/pass/set_softmax_ports.hpp"

#include "snippets/utils.hpp"

#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/assign_registers.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/lowered/pass/propagate_layout.hpp"
#include "snippets/lowered/pass/cleanup_loop_offsets.hpp"
#include "snippets/lowered/pass/softmax_decomposition.hpp"
#include "snippets/lowered/pass/move_scalar_to_consumer.hpp"
#include "snippets/lowered/pass/move_result_out_of_loop.hpp"
#include "snippets/lowered/pass/clean_repeated_ptr_shifts.hpp"
#include "snippets/lowered/pass/identify_buffers.hpp"
#include "snippets/lowered/pass/validate_loops.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"

#include "transformations/utils/utils.hpp"

#include <ngraph/pass/manager.hpp>
#include "ngraph/pass/constant_folding.hpp"
#include "ov_ops/type_relaxed.hpp"
#include <openvino/pass/serialize.hpp>

#include <algorithm>
#include <memory>
#include <array>

using namespace std;
using namespace ov::op::util;

namespace ov {
namespace snippets {

void snippets::op::Subgraph::set_generator(std::shared_ptr<ov::snippets::Generator> generator) {
    m_generator = generator;
}

void snippets::op::Subgraph::set_virtual_port_count(const size_t count) {
    m_virtual_port_count = count;
}

auto snippets::op::Subgraph::is_domain_sensitive_op(const std::shared_ptr<ov::Node>& op) -> bool {
    return ov::is_type<ov::op::v1::Transpose>(op) ||
           ov::is_type<ov::op::v1::Softmax>(op) ||
           ov::is_type<ov::op::v8::Softmax>(op) ||
           ov::is_type<ov::op::v0::MatMul>(op) ||
           ov::is_type<ov::op::v1::Broadcast>(op) || // Broadcast is domain sensetive op because the output shape depends on
           ov::is_type<ov::op::v3::Broadcast>(op);   // the both input and broadcast shapes (the both - are inputs of op). Note: is used only in MHA pattern
}

void snippets::op::Subgraph::init_config() {
    auto update = [](bool& flag, bool status) { flag = flag || status; };
    const auto ops = body_ptr()->get_ops();
    for (const auto& op : ops) {
        update(config.m_is_quantized, ov::is_type<ov::op::v0::FakeQuantize>(op));
        update(config.m_has_domain_sensitive_ops, is_domain_sensitive_op(op));
    }
}

auto snippets::op::Subgraph::get_estimated_buffer_count(const ov::NodeVector& ops) -> size_t {
    // The count of potential unique Buffers - it's hidden virtual ports as well
    // We should go through Subgraph and calculate potential non-inplace Buffers count.
    // These Buffers can be in 2 cases:
    // 1. Around Loops: we should check for element type size of nodes which use Buffer to get rating from above for unique Buffer count.
    // 2. Around MatMul: all buffers around Matmul must not be inplace because MatMul blocking implementation changes registers during computations.
    // The count is estimated because when we calculate this number, we have only original graph representation
    // and where will be Loops - we can just predict.
    // Note: The ops that create Buffers: MatMul, Transpose and Softmax (always FP32)
    std::vector<size_t> used_precision_size;

    auto push_prc_size = [&used_precision_size](size_t precision_size) {
        if (used_precision_size.empty() || used_precision_size.back() != precision_size) {
            used_precision_size.push_back(precision_size);
        }
    };

    for (const auto& op : ops) {
        if (const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(op)) {
            // At the moment Transposes are supported only on Results and Parameters but
            // then we should have the different Buffers for Transpose as well (Transpose isn't inplace)
            const auto consumers = transpose->get_output_target_inputs(0);
            // If after Transpose there is Result it means that there won't be Buffer after Transpose.
            // The same case is for Parameter before Transpose
            const auto are_prev_or_next_ops = std::none_of(consumers.begin(), consumers.end(),
                                                           [](const ov::Input<ov::Node>& in) {
                                                               return ov::is_type<ov::op::v0::Result>(in.get_node());
                                                           }) ||
                                              !ov::is_type<ov::op::v0::Parameter>(transpose->get_input_node_shared_ptr(0));
            if (are_prev_or_next_ops) {
                push_prc_size(transpose->get_element_type().size());
            }
        } else if (ov::is_type<ov::op::v1::Softmax>(op) || ov::is_type<ov::op::v8::Softmax>(op)) {
            // Softmax always uses 2 FP32 Buffers after decomposition.
            // They are inplace and the same so we can push precision size only once
            push_prc_size(ov::element::f32.size());
        } else if (const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op)) {
            // Since all buffers around Matmul must be unique, we explicitely add values to the vector without any checks
            if (!ov::is_type<ov::op::v0::Parameter>(matmul->get_input_node_shared_ptr(0)))
                used_precision_size.push_back(matmul->get_input_element_type(0).size());
            if (!ov::is_type<ov::op::v0::Parameter>(matmul->get_input_node_shared_ptr(1)))
                used_precision_size.push_back(matmul->get_input_element_type(1).size());

            const auto consumers = matmul->get_output_target_inputs(0);
            if (std::none_of(consumers.begin(), consumers.end(),
                             [](const ov::Input<ov::Node>& in) { return ov::is_type<ov::op::v0::Result>(in.get_node()); })) {
                used_precision_size.push_back(matmul->get_element_type().size());
            }
        }
    }

    return used_precision_size.size();
}

snippets::op::Subgraph::Subgraph(const OutputVector& args, std::shared_ptr<ov::Model> body)
    : SubGraphOp(args), m_generator(nullptr) {
    set_function(body);
    init_config();
    constructor_validate_and_infer_types();
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
    m_transformations_allowed = false;
}

snippets::op::Subgraph::Subgraph(const NodeVector& args, std::shared_ptr<ov::Model> body)
    : Subgraph(as_output_vector(args), std::move(body)) {}

std::shared_ptr<Node> snippets::op::Subgraph::clone_with_new_inputs(const OutputVector& inputs) const {
    INTERNAL_OP_SCOPE(Subgraph);
    return make_shared<Subgraph>(inputs, body().clone());
}

std::vector<PartialShape> snippets::op::Subgraph::reshape_body(const std::vector<PartialShape>& input_shapes) {
    auto& params = body_ptr()->get_parameters();
    OPENVINO_ASSERT(params.size() == input_shapes.size(), "Got invalid number of input shapes to reshape subgraph body");
    for (size_t i = 0; i < params.size(); ++i) {
        params[i]->set_partial_shape(input_shapes[i]);
    }
    body_ptr()->validate_nodes_and_infer_types();
    std::vector<PartialShape> output_shapes;
    for (const auto& res : body_ptr()->get_results()) {
        output_shapes.emplace_back(res->get_input_partial_shape(0));
    }
    return output_shapes;
}

std::vector<Shape> snippets::op::Subgraph::reshape_body(const std::vector<Shape>& input_shapes) {
    auto& params = body_ptr()->get_parameters();
    OPENVINO_ASSERT(params.size() == input_shapes.size(), "Got invalid number of input shapes to reshape subgraph body");
    for (size_t i = 0; i < params.size(); ++i) {
        params[i]->set_partial_shape(input_shapes[i]);
    }
    body_ptr()->validate_nodes_and_infer_types();
    std::vector<Shape> output_shapes;
    for (const auto& res : body_ptr()->get_results()) {
        auto pshape = res->get_input_partial_shape(0);
        OPENVINO_ASSERT(pshape.is_static(), "Subgraph inferred dynamic output shape during reshape with static inputs");
        output_shapes.emplace_back(res->get_input_partial_shape(0).get_shape());
    }
    return output_shapes;
}

void snippets::op::Subgraph::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::validate_and_infer_types")
    ov::ParameterVector old_parameters;
    for (auto op : body_ptr()->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        body_ptr()->replace_parameter(i, std::make_shared<ov::op::v0::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    body_ptr()->validate_nodes_and_infer_types();

    for (size_t i = 0; i < body_ptr()->get_parameters().size(); i++) {
        body_ptr()->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(body_ptr()->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, body_ptr()->get_output_element_type(i), body_ptr()->get_output_partial_shape(i));
    }
}

bool snippets::op::Subgraph::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("body", body_ptr());
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    return true;
}

auto snippets::op::Subgraph::wrap_node_as_subgraph(const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<op::Subgraph> {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::wrap_node_as_subgraph")
    ov::ParameterVector body_parameters;
    ov::OutputVector body_inputs;

    ov::OutputVector subgraph_inputs;

    for (const auto& input : node->input_values()) {
        if (ov::is_type<ov::opset1::Constant>(input.get_node_shared_ptr()) &&
            (ov::shape_size(input.get_shape()) == 1 ||
             ov::is_type<ov::op::v0::FakeQuantize>(node) ||
             constant_input_should_be_inside_body(node))) {
            body_inputs.push_back(input);
        } else {
            auto parameter = std::make_shared<ov::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            body_parameters.push_back(parameter);
            body_parameters.back()->set_friendly_name(input.get_node()->get_friendly_name());
            body_inputs.push_back(parameter->output(0));

            subgraph_inputs.push_back(input);
        }
    }

    auto body_node = node->clone_with_new_inputs(body_inputs);
    body_node->set_friendly_name(node->get_friendly_name());
    for (size_t i = 0; i < node->get_output_size(); i++) {
        fill_empty_output_names(body_node->output(i), node->output(i));
    }

    if (node->get_output_size() != body_node->get_output_size()) {
        OPENVINO_THROW("original node outputs size and extracted subgraph node outputs size doesn't much");
    }

    ov::ResultVector body_results;
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<ov::opset1::Result>(body_node->output(output.get_index())));
    }

    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, subgraph_inputs, body);

    size_t hidden_data_count = 0lu;
    if (auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node)) {
        hidden_data_count += utils::get_non_scalar_constant_count_for_fq(fq_node);
    }
    subgraph->set_virtual_port_count(hidden_data_count);

    for (size_t i = 0; i < body->get_parameters().size(); i++) {
        body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }

    if (subgraph->get_output_size() != body->get_results().size()) {
        OPENVINO_THROW("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

void snippets::op::Subgraph::fill_empty_output_names(const Output<Node>& target_output_node, const Output<Node>& replacement_output_node) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto& out_tensor = target_output_node.get_tensor();
    const std::string new_name = ov::op::util::get_ie_output_name(replacement_output_node);
    if (ov::descriptor::get_ov_tensor_legacy_name(out_tensor).empty()) {
        ov::descriptor::set_ov_tensor_legacy_name(out_tensor, new_name);
    }
    if (!replacement_output_node.get_names().empty()) {
        out_tensor.set_names(replacement_output_node.get_names());
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

auto snippets::op::Subgraph::constant_input_should_be_inside_body(const std::shared_ptr<ov::Node>& node) -> bool {
    return ov::is_type<ov::op::v1::Transpose>(node) ||
           ov::is_type<ov::op::v1::Broadcast>(node) ||
           ov::is_type<ov::op::v3::Broadcast>(node) ||
           ov::is_type<ov::op::v1::Reshape>(node);
}

///
/// \brief  Canonization transforms original subgraph and to canonical form suitable for code generation. In particular,
///         it handles supported layout conversions, broadcasts inputs and outputs to a single rank and layout. Canonicalization
///         returns master-shape (max rank + max dimensions over all outputs) that can be used for scheduling.
///         Canonicalization currently supports only the following layout conversions:
///             * None: all inputs have the same layout
///             * Planar + blocked: some inputs have blocked, and some have planar layouts, e.g. <N, C, H, W, c> + <N, C, H, W>
///         Also there is precision aligning inside body of subgraph during canonicalization
ov::PartialShape snippets::op::Subgraph::canonicalize(const BlockedShapeVector& outputShapes,
                                                      const BlockedShapeVector& inputShapes) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::canonicalize")
    NODE_VALIDATION_CHECK(this, inputShapes.size() == body_ptr()->get_parameters().size(),
        "Number of parameters for snippet doesn't match passed to generate method: ", inputShapes.size(), " vs ", body_ptr()->get_parameters().size(), ".");

    NODE_VALIDATION_CHECK(this, outputShapes.size() == body_ptr()->get_results().size(),
        "number of results for snippet doesn't match passed to generate method: ", outputShapes.size(), " vs ", body_ptr()->get_results().size(), ".");

    auto getMaxRankBlockedShape = [](const BlockedShapeVector& blockedShapes) -> const BlockedShape& {
        return *std::max_element(blockedShapes.begin(), blockedShapes.end(),
                         [&](const BlockedShape& lhs, const BlockedShape& rhs) {
                            return std::get<0>(lhs).size() < std::get<0>(rhs).size();
                         });
    };
    PartialShape baseShape;
    AxisVector baseOrder;
    std::tie(baseShape, baseOrder, std::ignore) = getMaxRankBlockedShape(inputShapes);
    const auto baseRank = baseShape.size();
    const bool baseIsBlocked = baseOrder.size() != std::set<size_t>(baseOrder.begin(), baseOrder.end()).size();
    for (size_t i = 0; i < inputShapes.size(); i++) {
        const auto& blockedShape = inputShapes[i];
        PartialShape inShape;
        AxisVector inOrder;
        element::Type inType;
        std::tie(inShape, inOrder, inType) = blockedShape;
        const auto inRank = inShape.size();
        NODE_VALIDATION_CHECK(this, inRank <= baseRank, "Input rank can't be larger than output rank in snippets.");
        if (inRank < baseRank) {
            PartialShape newShape(ov::Shape(baseRank, 1));
            // todo: more complicated logics is needed if we want to merge smth else than blocked and planar
            if (baseIsBlocked) {
                const bool inIsNotBlocked = inOrder.size() == std::set<size_t>(inOrder.begin(), inOrder.end()).size();
                NODE_VALIDATION_CHECK(this, inIsNotBlocked, "Snippets don't support conversion between blocked layouts of different ranks");
                inShape.insert(inShape.end(), ov::Dimension(1));
            }
            NODE_VALIDATION_CHECK(this, PartialShape::broadcast_merge_into(newShape, inShape, ov::op::AutoBroadcastType::NUMPY),
                                  "Failed to broadcast_merge inputs in snippets canonicalization");
            inShape = std::move(newShape);
        } else {
            // todo: 4d blocked + 5d planar layouts are not supported: <N, C, H, W, c> + <N, C, D, H, W>
            NODE_VALIDATION_CHECK(this,
                                  equal(baseOrder.begin(), baseOrder.end(), inOrder.begin()),
                                  "Snippets canonicalization got input shapes of equal ranks but different layouts, which is not supported");
        }
        ov::PartialShape tmpPShape(baseShape);
        // todo: we need to generalize canonicalization for domain-sensitive ops. E.g. MatMul inputs can't be broadcasted one to another
        if (!config.m_has_domain_sensitive_ops)
            NODE_VALIDATION_CHECK(this,
                                  PartialShape::broadcast_merge_into(tmpPShape, inShape, ::ov::op::AutoBroadcastType::NUMPY),
                                  "Failed to create broadcastable shapes in snippets canonicalization");
        const auto paramShape = body_ptr()->get_parameters()[i]->get_partial_shape();
        const auto paramType =  body_ptr()->get_parameters()[i]->get_element_type();
        if (paramShape.size() != inShape.size() || !equal(paramShape.begin(), paramShape.end(), inShape.begin()))
                body_ptr()->replace_parameter(i, std::make_shared<ov::op::v0::Parameter>(paramType, inShape));
    }
    body_ptr()->validate_nodes_and_infer_types();
    auto skipStartEndOnes = [](const PartialShape& shape) {
        auto begin = shape.begin();
        auto end = shape.end();
        while (begin != end && *begin == 1)
            begin++;
        while (begin != end && *(end-1) == 1)
            end--;

        PartialShape trimmedShape(std::vector<ov::Dimension> (end - begin, 1));
        std::copy(begin, end, trimmedShape.begin());
        return trimmedShape;
    };

    // Check that output shapes are broadcastable => can be scheduled
    const auto& body_results = body_ptr()->get_results();
    PartialShape outPShape = body_results[0]->get_input_partial_shape(0);
    // todo: we need a slightly more general approach for backward ROI propagation
    const auto& result_parent = body_results[0]->get_input_node_shared_ptr(0);
    if (body_results.size() == 1 &&
        ov::is_type<ov::op::v1::Transpose>(result_parent) &&
        ov::is_type<ov::op::v0::MatMul>(result_parent->get_input_node_shared_ptr(0))) {
        outPShape = result_parent->get_input_partial_shape(0);
    } else {
        for (size_t i = 0; i < body_results.size(); i++) {
            auto shape_i = body_results[i]->get_input_partial_shape(0);
            auto outputShape_i = std::get<0>(outputShapes[i]);
            // Check that the produced output shape corresponds to the passed shape
            // Some produced shapes may have been changed to be broadcastable (e.g. blocked + planar outputs),
            // so we need to remove leading and trailing "1" before the comparison
            PartialShape pShape_i(skipStartEndOnes(shape_i));
            bool compatibleWithPassedShape = PartialShape::broadcast_merge_into(pShape_i,
                                                                                skipStartEndOnes(outputShape_i),
                                                                                ::ov::op::AutoBroadcastType::NUMPY);
            NODE_VALIDATION_CHECK(this, compatibleWithPassedShape,
                                  "Inferred and passed results shapes are incompatible for snippet ");
            // Check that output shapes are broadcastable to each other => can be scheduled
            bool compatibleWithOtherOutputs = PartialShape::broadcast_merge_into(outPShape, shape_i,
                                                                                 ::ov::op::AutoBroadcastType::NUMPY);
            NODE_VALIDATION_CHECK(this, compatibleWithOtherOutputs,
                                  "Snippets output shapes must be numpy broadcastable");
        }
    }

    // We should insert Converts after Parameters and Constant and before Results
    // to align precision inside Subgraph body that is supported by Plugin
    align_element_types(outputShapes, inputShapes);

    master_shape = outPShape;
    return master_shape;
}

bool snippets::op::Subgraph::check_broadcast(const std::shared_ptr<const ov::Node>& node) noexcept {
    const auto elementwise = std::dynamic_pointer_cast<const ov::op::util::BinaryElementwiseArithmetic>(node);
    return
        (elementwise == nullptr) ||
        (elementwise->get_input_partial_shape(0).size() == elementwise->get_input_partial_shape(1).size()) ||
        (elementwise->get_autob().m_type != ov::op::AutoBroadcastType::PDPD);
}

void snippets::op::Subgraph::align_element_types(const BlockedShapeVector& outputShapes,
                                                 const BlockedShapeVector& inputShapes) {
    // We should insert Convert before Results to set original output element type if needed
    const auto& body_results = body_ptr()->get_results();
    for (size_t i = 0; i < outputShapes.size(); i++) {
        const auto needed_out_type = std::get<2>(outputShapes[i]);
        if (body_results[i]->get_input_element_type(0) != needed_out_type) {
            const auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(
                body_results[i]->get_input_node_shared_ptr(0), needed_out_type);
            body_results[i]->set_argument(0, convert);
            body_results[i]->validate_and_infer_types();
        }
    }

    // We should change existing element type to original for Parameters if needed
    const auto& parameters = body_ptr()->get_parameters();
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        const auto needed_in_type = std::get<2>(inputShapes[i]);
        const auto& parameter = parameters[i];
        if (parameter->get_element_type() != needed_in_type) {
            const auto parameter_output = parameter->output(0);
            const auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(
                parameter_output,
                parameter_output.get_element_type());
            ov::copy_runtime_info(parameter, convert);

            for (const auto input : parameter_output.get_target_inputs()) {
                const auto& input_node = input.get_node();
                if (input_node == convert.get()) {
                    continue;
                }
                input_node->set_argument(input.get_index(), convert->output(0));
            }

            parameter->set_element_type(needed_in_type);
            parameter->validate_and_infer_types();
        }
    }
}

void snippets::op::Subgraph::data_flow_transformations(ov::pass::Manager& pre_common,
                                                       ov::pass::Manager& post_common,
                                                       ov::pass::Manager& post_precision) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::data_flow_transformations")

    const auto&  params = body_ptr()->get_parameters();
    bool inputs_has_dynamic_last_dims = std::any_of(params.begin(), params.end(),
                                                    [](const shared_ptr<ov::op::v0::Parameter>& p) {
                                                        return p->get_partial_shape().rbegin()->is_dynamic();
                                                    });

    pre_common.run_passes(body_ptr());

    ov::pass::Manager common_manager;
    if (config.m_has_domain_sensitive_ops) {
        common_manager.register_pass<snippets::pass::MatMulToBrgemm>();
        common_manager.register_pass<snippets::pass::FuseTransposeBrgemm>();
        common_manager.register_pass<snippets::pass::TransposeDecomposition>();
        common_manager.register_pass<snippets::pass::SetSoftmaxPorts>();
    }
    common_manager.register_pass<snippets::pass::BroadcastToMoveBroadcast>();
    common_manager.register_pass<snippets::pass::ConvertConstantsToScalars>();
    common_manager.register_pass<snippets::pass::ConvertPowerToPowerStatic>();
    // todo: presently dynamic pipeline is activated even if the last two dimension are static
    //  In general, we can use static kernels in this case, but several parameters (src and dst memory pointers for example)
    //  should be passed as run-time args, so it's a mixed mode: kernel is shape-aware, but some additional runtime args are required
    // Presently Broadcasting is organized in the following way:
    // * ALL last dims are static => broadcasting is handled via MoveBroadcast and pointer arithmetics (even for dynamic upper dims)
    if (!inputs_has_dynamic_last_dims) {
        common_manager.register_pass<snippets::pass::InsertMoveBroadcast>();
    }
    common_manager.run_passes(body_ptr());

    post_common.run_passes(body_ptr());

    ov::pass::Manager precision_manager;
    precision_manager.register_pass<snippets::pass::PropagatePrecision>(m_generator->get_target_machine());
    precision_manager.register_pass<ov::pass::ConstantFolding>();
    precision_manager.register_pass<snippets::pass::ConvertConstantsToScalars>();
    precision_manager.run_passes(body_ptr());

    post_precision.run_passes(body_ptr());
}

void snippets::op::Subgraph::control_flow_transformations(lowered::LinearIR& linear_ir,
                                                          lowered::pass::PassPipeline& target_markup_pipeline,
                                                          lowered::pass::PassPipeline& target_pipeline) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::control_flow_transformations")

    const size_t vector_size = get_generator()->get_target_machine()->get_lanes();
    const int32_t buffer_allocation_rank = static_cast<int32_t>(linear_ir.get_config().m_loop_depth);

    // Ticket: 113666
    // TODO: Make pass pipeline with backend passes more flexible
    target_markup_pipeline.run(linear_ir);

    lowered::pass::PassPipeline common_pipeline;
    common_pipeline.register_pass<lowered::pass::MarkLoops>(vector_size);
    common_pipeline.register_pass<lowered::pass::SoftmaxDecomposition>(vector_size);
    common_pipeline.register_pass<lowered::pass::FuseLoops>();
    common_pipeline.register_pass<lowered::pass::SplitLoops>();
    common_pipeline.register_pass<lowered::pass::MoveResultOutOfLoop>();
    common_pipeline.register_pass<lowered::pass::InsertBuffers>(buffer_allocation_rank);
    common_pipeline.register_pass<lowered::pass::InsertLoadStore>(vector_size);
    common_pipeline.register_pass<lowered::pass::MoveScalarToConsumer>();
    common_pipeline.register_pass<lowered::pass::LoadMoveBroadcastToBroadcastLoad>();
    common_pipeline.register_pass<lowered::pass::ValidateLoops>();
    common_pipeline.register_pass<lowered::pass::InitLoops>();
    common_pipeline.register_pass<lowered::pass::InsertLoops>();
    common_pipeline.run(linear_ir);

    target_pipeline.run(linear_ir);

    const auto buffer_allocation_pass = std::make_shared<lowered::pass::AllocateBuffers>();
    lowered::pass::PassPipeline buffer_pipeline;
    buffer_pipeline.register_pass<lowered::pass::IdentifyBuffers>();
    buffer_pipeline.register_pass<lowered::pass::CleanRepeatedDataPointerShifts>();
    buffer_pipeline.register_pass(buffer_allocation_pass);
    buffer_pipeline.run(linear_ir);

    lowered::pass::PassPipeline final_pipeline;
    final_pipeline.register_pass<lowered::pass::PropagateLayout>();
    final_pipeline.register_pass<lowered::pass::CleanupLoopOffsets>();
    final_pipeline.run(linear_ir);

    m_buffer_scratchpad = buffer_allocation_pass->get_scratchpad_size();
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    const void* compile_params) {
    canonicalize(output_shapes, input_shapes);
    return generate(compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(const BlockedShapeVector& output_shapes,
                                                    const BlockedShapeVector& input_shapes,
                                                    ov::pass::Manager& pre_common,
                                                    ov::pass::Manager& post_common,
                                                    ov::pass::Manager& post_precision,
                                                    lowered::pass::PassPipeline& target_lowered_markup_pipeline,
                                                    lowered::pass::PassPipeline& target_lowered_pipeline,
                                                    const void* compile_params) {
    canonicalize(output_shapes, input_shapes);
    return generate(pre_common, post_common, post_precision, target_lowered_markup_pipeline, target_lowered_pipeline, compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(const void* compile_params) {
    auto mngr = ov::pass::Manager();
    auto lowered = lowered::pass::PassPipeline();
    return generate(mngr, mngr, mngr, lowered, lowered, compile_params);
}

snippets::Schedule snippets::op::Subgraph::generate(
    ov::pass::Manager& pre_common,
    ov::pass::Manager& post_common,
    ov::pass::Manager& post_precision,
    lowered::pass::PassPipeline& target_lowered_markup_pipeline,
    lowered::pass::PassPipeline& target_lowered_pipeline,
    const void* compile_params) {
    INTERNAL_OP_SCOPE(Subgraph);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::generate")
    NGRAPH_CHECK(m_generator != nullptr, "generate is called while generator is not set");

    data_flow_transformations(pre_common, post_common, post_precision);

    lowered::Config lowering_config;
    lowering_config.m_save_expressions = config.m_has_domain_sensitive_ops;
    lowering_config.m_need_fill_tail_register = config.m_has_domain_sensitive_ops;
    lowering_config.m_loop_depth = tileRank;

    lowered::LinearIR linear_ir = lowered::LinearIR(body_ptr(), lowering_config);
    control_flow_transformations(linear_ir, target_lowered_markup_pipeline, target_lowered_pipeline);

    // actual code emission
    const auto& lowering_result = m_generator->generate(linear_ir, lowering_config, compile_params);
    const auto ptr = lowering_result.binary_code;

    return {master_shape, false /*canBeLinearized*/, ptr};
}

void snippets::op::Subgraph::print() const {
    INTERNAL_OP_SCOPE(Subgraph);
    remark(13) << "subgraph " << this->get_friendly_name() << " "
        << this->get_type_name()
        << " which contains " << body_ptr()->get_ops().size() << " nodes" << std::endl;

    int qqq = 0;
    for (auto op : body_ptr()->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }

    for (auto& in : this->inputs()) {
        remark(13) << "  -> " << in.get_source_output().get_node_shared_ptr()->get_friendly_name() << " "
            << in.get_source_output().get_node_shared_ptr() << std::endl;
    }

    for (auto& out : this->outputs()) {
        for (auto& user : out.get_target_inputs()) {
            remark(13) << " <- " << user.get_node()->get_friendly_name() << " "  << user.get_node() << std::endl;
        }
        remark(13) << std::endl;
    }
}

void snippets::op::Subgraph::print_statistics(bool verbose) {
    INTERNAL_OP_SCOPE(Subgraph);
    auto getNodeInventory = [](std::shared_ptr<ov::Node> n) -> size_t {
        size_t total = 0;

        for (auto input : n->inputs()) {
            total += input.get_tensor().size();
        }

        for (auto output : n->outputs()) {
            total += output.get_tensor().size();
        }

        if (auto subgraph = ov::as_type_ptr<op::Subgraph>(n)) {
            for (auto op : subgraph->body_ptr()->get_ordered_ops()) {
                if (ov::as_type_ptr<ov::opset1::Constant>(op)) {
                    total += op->output(0).get_tensor().size();
                }
            }
        }

        return total;
    };

    auto getModelInventory = [getNodeInventory](const ov::Model & f) -> size_t {
        size_t total = 0;
        for (auto op : f.get_ordered_ops()) {
            // Results and parameters are artificially introduced,
            // while Constants are already considered if they are inputs of other operation
            // this should lead to 1:1 inventory for single node operations
            if (!ov::as_type_ptr<ov::opset1::Parameter>(op)
             && !ov::as_type_ptr<ov::opset1::Result>(op)
             && !ov::as_type_ptr<ov::opset1::Constant>(op)) {
                total += getNodeInventory(op);
            }
        }
        return total;
    };

    auto countConstants = [](const ov::Model & f) -> size_t {
        size_t count = 0;
        for (auto op : f.get_ordered_ops()) {
            count += !!ov::as_type_ptr<ov::opset1::Constant>(op) ? 1 : 0;
        }
        return count;
    };

    std::cout << get_friendly_name()
                << ";" << this
                << ";" << body_ptr()->get_ops().size()
                << ";" << body_ptr()->get_parameters().size()
                << ";" << body_ptr()->get_results().size()
                << ";" << countConstants(body())
                << ";" << getModelInventory(body())
                << ";" << getNodeInventory(shared_from_this()) << std::endl;

    if (verbose) {
        this->print();
    }
}

void snippets::op::Subgraph::serialize() const {
    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, xmlFile, ov::pass::Serialize::Version::IR_V10);
    serializer.run_on_model(body_ptr());
    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();
    std::cout << m_model << std::endl;
}

} // namespace snippets
} // namespace ov
