// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <vector>

#include "ngraph/deprecated.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
namespace op {
namespace util {
///
/// \brief      Validates static rank and dimension for provided input parameters.
///             Additionally input_size dimension is checked for X and W inputs.
///             Applies to LSTM, GRU and RNN Sequences.
///
///
/// \param[in]  input        Vector with RNNSequence-like op inputs in following order:
///                          X, initial_hidden_state, sequence_lengths, W, R and B.
///
NGRAPH_API_DEPRECATED void validate_seq_input_rank_dimension(const std::vector<ngraph::PartialShape>& input);
}  // namespace util
}  // namespace op
}  // namespace ngraph
