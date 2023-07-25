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

class TRANSFORMATIONS_API ConvertGather0D;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather0D decomposes v1::Gather operation into v0::Unsqueeze + v1::Gather + v0::Squeeze pattern when
 * gather indices is scalar
 */
class ov::pass::ConvertGather0D : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGather0D", "0");
    ConvertGather0D();
};
