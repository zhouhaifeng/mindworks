// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "low_precision/reduce_base_transformation.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief ReduceMaxTransformation propagates dequantization operations through ReduceMax operation.
 *
 * For more details about the transformation, refer to
 * [ReduceMaxTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMaxTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API ReduceMaxTransformation : public ReduceBaseTransformation {
public:
    OPENVINO_RTTI("ReduceMaxTransformation", "0");
    ReduceMaxTransformation(const Params& params = Params());
    bool isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const override;

protected:
    bool getUpdatePrecision(const std::shared_ptr<Node>& reduce) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
