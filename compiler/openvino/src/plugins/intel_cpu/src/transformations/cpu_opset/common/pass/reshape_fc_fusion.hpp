// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ReshapeFullyConnectedFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeFullyConnectedFusion", "0");
    ReshapeFullyConnectedFusion();
};

}   // namespace intel_cpu
}   // namespace ov
