// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertMulOrAddFinally;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulOrAddFinally : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertMulOrAddFinally", "0");
    ConvertMulOrAddFinally();
};
