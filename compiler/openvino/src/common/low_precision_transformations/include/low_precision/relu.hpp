// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief ReluTransformation propagates dequantization operations through Relu operation.
 *
 * For more details about the transformation, refer to
 * [ReluTransformation](@ref openvino_docs_OV_UG_lpt_ReluTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API ReluTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("ReluTransformation", "0");
    ReluTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
