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

class TRANSFORMATIONS_API ConvertSubtract;
class TRANSFORMATIONS_API ConvertSubtractWithConstant;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertSubtract : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSubtract", "0");
    ConvertSubtract();
};

class ov::pass::ConvertSubtractWithConstant : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSubtractWithConstant", "0");
    ConvertSubtractWithConstant();
};
