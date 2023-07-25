// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_sequences_optimization.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

namespace {
    int64_t getSeqAxis(const std::shared_ptr<ngraph::Node>& sequenceOp) {
        // Optimization.
        // Plug-ins support seqAxis attribute (value 1 or 0) for Seq ops, but according to the spec we don't
        // support this attribute and should insert Transpose layer before and after Seq op in TI to Sequences
        // transformation. Additional Transpose layers affect the performance, so we try to detect pattern
        // Transpose(axis_order={1,0,2}) -> Seq -> Transpose(axis_order={2,1,0,3}
        // and replace unnecessary Transpose ops with SeqIE (seqAxis = 0) to transfer value
        // of the attribute to plug-ins.
        // todo: specify seqAxis attribute for Sequence ops.
        int64_t seqAxis = 1; // default
        const auto& target_inputs = sequenceOp->get_output_target_inputs(0);
        if (target_inputs.size() == 1) {
            const auto transpose_before = ngraph::as_type_ptr<ngraph::opset1::Transpose>(sequenceOp->get_input_node_shared_ptr(0));
            const auto transpose_after = ngraph::as_type_ptr<ngraph::opset1::Transpose>(target_inputs.begin()->get_node()->shared_from_this());

            if (transpose_after && transpose_before) {
                auto order_before = ngraph::as_type_ptr<ngraph::opset1::Constant>(transpose_before->get_input_node_shared_ptr(1));
                auto order_after = ngraph::as_type_ptr<ngraph::opset1::Constant>(transpose_after->get_input_node_shared_ptr(1));

                if (order_before && order_after) {
                    auto order_before_values = order_before->cast_vector<int64_t>();
                    auto order_after_values = order_after->cast_vector<int64_t>();
                    std::vector<int64_t> order_ref_before = {1, 0, 2};
                    std::vector<int64_t> order_ref_after = {2, 1, 0, 3};
                    if (order_before_values == order_ref_before && order_after_values == order_ref_after) {
                        seqAxis = 0;
                    }
                }
            }
        }
        return seqAxis;
    }

    bool transform(const std::shared_ptr<ngraph::Node>& sequenceOp) {
        // Detect pattern: Transpose_before -> Seq -> Transpose_after
        auto seqAxis = getSeqAxis(sequenceOp);
        if (seqAxis == 0) {
            ngraph::Output<ngraph::Node> in_0 = sequenceOp->get_input_node_shared_ptr(0)->input_value(0);

            auto shapeBeforeTranspose = ov::op::util::make_try_fold<ngraph::opset1::ShapeOf>(in_0);
            auto newInShape = ov::op::util::make_try_fold<ngraph::opset8::Gather>(shapeBeforeTranspose,
                ngraph::opset1::Constant::create(ngraph::element::i32, { 3 }, { 1, 0, 2 }),
                ngraph::opset1::Constant::create(ngraph::element::i32, {}, { 0 }));
            auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(in_0, newInShape, false);
            ngraph::copy_runtime_info(sequenceOp->get_input_node_shared_ptr(0), reshape1);
            ngraph::replace_node(sequenceOp->get_input_node_shared_ptr(0), reshape1);

            const auto &seqTargetInputs = sequenceOp->get_output_target_inputs(0);
            if (seqTargetInputs.empty())
                return false;
            auto transposeAfter = seqTargetInputs.begin()->get_node()->shared_from_this();

            auto lstmOutShape = ov::op::util::make_try_fold<ngraph::opset1::ShapeOf>(sequenceOp->output(0));
            auto newOutShape = ov::op::util::make_try_fold<ngraph::opset8::Gather>(lstmOutShape,
                ngraph::opset1::Constant::create(ngraph::element::i32, { 4 }, { 2, 1, 0, 3 }),
                ngraph::opset1::Constant::create(ngraph::element::i32, {}, { 0 }));

            auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(sequenceOp->output(0), newOutShape, false);
            reshape2->set_friendly_name(transposeAfter->get_friendly_name());
            ngraph::copy_runtime_info(transposeAfter, reshape2);
            ngraph::replace_node(transposeAfter, reshape2);
        }

        sequenceOp->get_rt_info()["seqAxis"] = seqAxis;

        return true;
    }
} // namespace

ov::intel_cpu::OptimizeGRUSequenceTransposes::OptimizeGRUSequenceTransposes() {
    MATCHER_SCOPE(OptimizeGRUSequenceTransposes);
    auto gruSequenceNgraph = ngraph::pattern::wrap_type<ngraph::opset5::GRUSequence>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto gruSequence = ngraph::as_type_ptr<ngraph::opset5::GRUSequence>(m.get_match_root());
        if (!gruSequence) {
            return false;
        }
        // Bidirectional cases are not supported
        if (gruSequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        return transform(gruSequence);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gruSequenceNgraph, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::OptimizeRNNSequenceTransposes::OptimizeRNNSequenceTransposes() {
    MATCHER_SCOPE(OptimizeRNNSequenceTransposes);
    auto rnnSequenceNgraph = ngraph::pattern::wrap_type<ngraph::opset5::RNNSequence>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto rnnSequence = ngraph::as_type_ptr<ngraph::opset5::RNNSequence>(m.get_match_root());
        if (!rnnSequence) {
            return false;
        }
        // Bidirectional cases are not supported
        if (rnnSequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        return transform(rnnSequence);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnnSequenceNgraph, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::OptimizeLSTMSequenceTransposes::OptimizeLSTMSequenceTransposes() {
    MATCHER_SCOPE(OptimizeLSTMSequenceTransposes);
    auto lstmSequenceNgraph = ngraph::pattern::wrap_type<ngraph::opset1::LSTMSequence, ngraph::opset5::LSTMSequence>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto checkSequence = [](const std::shared_ptr<ngraph::Node>& node) {
            if (auto lstm5 = ngraph::as_type_ptr<ngraph::opset5::LSTMSequence>(node)) {
                return lstm5->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL;
            } else if (auto lstm1 = ngraph::as_type_ptr<ngraph::opset1::LSTMSequence>(node)) {
                return lstm1->get_direction() != ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL;
            } else {
                return false;
            }
        };

        std::shared_ptr<ngraph::Node> lstmSequence = m.get_match_root();
        return checkSequence(lstmSequence) ? transform(lstmSequence) : false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstmSequenceNgraph, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::OptimizeSequenceTransposes::OptimizeSequenceTransposes() {
    ADD_MATCHER_FOR_THIS(OptimizeLSTMSequenceTransposes)
    ADD_MATCHER_FOR_THIS(OptimizeRNNSequenceTransposes)
    ADD_MATCHER_FOR_THIS(OptimizeGRUSequenceTransposes)
}
