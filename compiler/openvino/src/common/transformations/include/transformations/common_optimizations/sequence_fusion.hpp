// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SequenceFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SequenceFusion transformation replaces a chain of Cells
 * operations with single Sequence op.
 *
 * Supported cells: GRUCell, LSTMCell, RNNCell, AUGRUCell
 * Prerequisites: the source of W,R,B inputs must be the same or
 * it can be different Constants with the same type, shape and value.
 */

class ov::pass::SequenceFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SequenceFusion", "0");
    SequenceFusion();
};
