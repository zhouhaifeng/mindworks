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

class TRANSFORMATIONS_API SpaceToBatchFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SpaceToBatchFusion transformation replaces following graph:
 * Transpose (or Reshape) -> Pad -> SpaceToDepth -> Transpose (or Reshape)
 * to SpaceToBatch
 * Restrictions:
 * - input rank must be 4
 * - Transpose permutation must be [1, 0, 2, 3]
 * - pad value is 0, PadMode is CONSTANT
 * - SpaceToDepthMode must be BLOCKS_FIRST
 */

class ov::pass::SpaceToBatchFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SpaceToBatchFusion", "0");
    SpaceToBatchFusion();
};
