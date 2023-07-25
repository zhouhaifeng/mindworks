// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EyeDecomposition;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 *
 * @brief Do eye decomposition to sub-graph (model).
 */
class ov::pass::EyeDecomposition : public MatcherPass {
public:
    OPENVINO_RTTI("EyeDecomposition", "0");
    EyeDecomposition();
};
