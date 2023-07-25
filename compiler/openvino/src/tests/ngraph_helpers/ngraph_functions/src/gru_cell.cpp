// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeGRU(const OutputVector& in,
                                      const std::vector<ov::Shape>& constants,
                                      std::size_t hidden_size,
                                      const std::vector<std::string>& activations,
                                      const std::vector<float>& activations_alpha,
                                      const std::vector<float>& activations_beta,
                                      float clip,
                                      bool linear_before_reset,
                                      bool make_sequence,
                                      ov::op::RecurrentSequenceDirection direction,
                                      ngraph::helpers::SequenceTestsMode mode) {
    std::vector<float> empty;
    auto W = ngraph::builder::makeConstant(in[0].get_element_type(), constants[0], empty, true);
    auto R = ngraph::builder::makeConstant(in[0].get_element_type(), constants[1], empty, true);
    auto B = ngraph::builder::makeConstant(in[0].get_element_type(), constants[2], empty, true);
    if (!make_sequence) {
        return std::make_shared<ov::op::v3::GRUCell>(in[0], in[1], W, R, B, hidden_size, activations,
                                                         activations_alpha, activations_beta, clip,
                                                         linear_before_reset);
    } else {
        if (in.size() > 2 && in[2].get_partial_shape().is_dynamic()) {
            return std::make_shared<ov::op::v5::GRUSequence>(in[0], in[1], in[2], W, R, B, hidden_size, direction,
                                                                activations, activations_alpha, activations_beta, clip, linear_before_reset);
        } else {
            std::shared_ptr<Node> seq_lengths;
            switch (mode) {
                case ngraph::helpers::SequenceTestsMode::PURE_SEQ:
                case ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST: {
                    std::vector<float> lengths(in[0].get_partial_shape()[0].get_min_length(), in[0].get_partial_shape()[1].get_min_length());
                    seq_lengths = ngraph::builder::makeConstant(element::i64, constants[3], lengths, false);
                    break;
                }
                case ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST:
                case ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST: {
                    for (size_t i = 0; i <= in[0].get_shape().at(0); ++i) {
                        std::vector<float> lengths;
                        seq_lengths = ngraph::builder::makeConstant(element::i64, constants[3], lengths, true,
                                                                    static_cast<float>(in[0].get_shape()[1]), 0.f);
                    }
                    break;
                }
                case ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM:
                case ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM:
                case ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM: {
                    // Seq_lengths should be as a Parameter node for these two modes
                    seq_lengths = in.at(2).get_node_shared_ptr();
                    break;
                }
                default:
                    throw std::runtime_error("Incorrect mode for creation of Sequence operation");
            }
            return std::make_shared<ov::op::v5::GRUSequence>(in[0], in[1], seq_lengths, W, R, B, hidden_size, direction,
                                                                 activations, activations_alpha, activations_beta, clip, linear_before_reset);
        }
    }
}
}  // namespace builder
}  // namespace ngraph
