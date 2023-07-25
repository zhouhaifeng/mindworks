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

class TRANSFORMATIONS_API ReduceL1Decomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Decomposes ReduceL1 into ReduceSum(abs(x)).
 */
class ov::pass::ReduceL1Decomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceL1Decomposition", "0");
    ReduceL1Decomposition();
};
