// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertInterpolate1ToInterpolate4;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertInterpolate1ToInterpolate4 covert v0:interpolate into v4::Interpolate.
 */
class ov::pass::ConvertInterpolate1ToInterpolate4 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertInterpolate1ToInterpolate4", "0");
    ConvertInterpolate1ToInterpolate4();
};
