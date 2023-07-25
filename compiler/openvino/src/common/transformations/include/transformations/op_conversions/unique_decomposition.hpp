// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API UniqueDecomposition;

}  // namespace pass
}  // namespace ov

// This transformation expresses Unique with a sub-graph of OpenVINO operations
class ov::pass::UniqueDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("UniqueDecomposition", "0");
    UniqueDecomposition();
};
