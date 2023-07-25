﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transparent_base_transformation.hpp"

#include <memory>
#include <vector>

#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool TransparentBaseTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<Node> op = m.get_match_root();
    if (!canBeTransformed(context, op)) {
        return false;
    }

    op = NetworkHelper::separateInStandaloneBranch(op, defaultPrecisions);
    moveDequantizationAfter(context, op, NetworkHelper::getDequantization(op, defaultPrecisions), true);
    return true;
}

bool TransparentBaseTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return true;
}

bool TransparentBaseTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
