// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <low_precision/lpt_visibility.hpp>
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "openvino/pass/pass.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API PropagatePrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief PropagatePrecisions transformation propagates PrecisionsAttribute attribute instances precision preserved operations.
 *
 * For more details about the transformation, refer to
 * [PropagatePrecisions](@ref openvino_docs_OV_UG_lpt_PropagatePrecisions) page
 * in the Inference Engine Developer Guide.
 */
class ngraph::pass::low_precision::PropagatePrecisions : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("PropagatePrecisions", "0");
    PropagatePrecisions(const AttributeParameters& params = AttributeParameters());
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    const AttributeParameters params;
};
