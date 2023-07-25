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

class TRANSFORMATIONS_API WrapInterpolateIntoTransposes;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief WrapInterpolateIntoTransposes transformation replaces
 *      Interpolate
 *  with
 *      Transpose -> Interpolate -> Transpose
 *  when
 *    1) the source Interpolate has the static input rank;
 *    2) 'axes' input is a Constant;
 *    3) number of axes is equal to input rank minus 2;
 *    4) axes contain 0 or 1.
 *  The reason of this transformation is that now CPU plugin supports interpolation only
 *  with respect to spatial dimensions, but TensorFlow frontend gives Interpolate with
 *  axes {1, 2} for 4D tensors.
 */
class ov::pass::WrapInterpolateIntoTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("WrapInterpolateIntoTransposes", "0");
    WrapInterpolateIntoTransposes();
};
