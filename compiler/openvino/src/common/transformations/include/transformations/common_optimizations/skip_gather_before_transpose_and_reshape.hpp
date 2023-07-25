// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SkipGatherBeforeTransposeAndReshape;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SkipGatherBeforeTransposeAndReshape transformation removes Gather from the Gather->Transpose->Reshape sequence
 * in case when input has batch=1 and gather has axis=0 and indices={0}.
 * Also, this transformation corrects a transpose constant to save semantic.
 */
class ov::pass::SkipGatherBeforeTransposeAndReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SkipGatherBeforeTransposeAndReshape", "0");
    SkipGatherBeforeTransposeAndReshape();
};
