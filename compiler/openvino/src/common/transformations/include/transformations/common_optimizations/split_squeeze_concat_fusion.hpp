// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SplitSqueezeConcatFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SplitSqueezeConcatFusion transformation replaces group of
 * operations: Split -> Squeeze (multiple) -> Concat to Transpose -> Reshape ops.
 */
class ov::pass::SplitSqueezeConcatFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitSqueezeConcatFusion", "0");
    SplitSqueezeConcatFusion();
};
