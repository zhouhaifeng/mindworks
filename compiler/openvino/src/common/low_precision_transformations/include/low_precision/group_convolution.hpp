// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "convolution.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief GroupConvolutionTransformation propagates dequantization operations through GroupConvolution operation.
 *
 * For more details about the transformation, refer to
 * [GroupConvolutionTransformation](@ref openvino_docs_OV_UG_lpt_GroupConvolutionTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API GroupConvolutionTransformation : public ConvolutionTransformation {
public:
    OPENVINO_RTTI("GroupConvolutionTransformation", "0");
    GroupConvolutionTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isQuantized(const std::shared_ptr<const Node>& layer,
        const std::vector<ngraph::element::Type>& defaultPrecisions) const override;
    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer,
        const std::vector<ngraph::element::Type>& defaultPrecisions);

protected:
    size_t getInputChannels(const std::shared_ptr<ngraph::Node> conv) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
