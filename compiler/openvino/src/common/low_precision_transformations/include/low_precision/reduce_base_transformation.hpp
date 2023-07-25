// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief ReduceBaseTransformation: base class for Reduce*Transformation,
 * detects dequantization operations in front of the Reduce* operation and
 * propagates them through the Reduce* if possible.
 */

class LP_TRANSFORMATIONS_API ReduceBaseTransformation : public LayerTransformation {
public:
    ReduceBaseTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const override;

protected:
    virtual void changeDequantizationValues(
        const std::shared_ptr<Node>& reduce,
        FakeQuantizeDequantization& dequantization) const;
    virtual bool getUpdatePrecision(const std::shared_ptr<Node>& reduce) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
