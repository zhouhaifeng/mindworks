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

class TRANSFORMATIONS_API DilatedConvolutionConverter;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief DilatedConvolutionConverter transformation replaces following graph:
 * SpaceToBatch -> Convolution(GroupConvolution) -> BatchToSpace
 * to a single Convolution(GroupConvolution) node with updated pads and dilations
 * Restrictions:
 * - pads in SpaceToBatch must have 0 on first and second position
 */

class ov::pass::DilatedConvolutionConverter : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DilatedConvolutionConverter", "0");
    DilatedConvolutionConverter();
};
