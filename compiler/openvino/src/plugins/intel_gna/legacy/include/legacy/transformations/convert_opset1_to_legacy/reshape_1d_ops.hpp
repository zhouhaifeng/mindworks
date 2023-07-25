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

class Reshape1DOps;
class Reshape1DConvolution;
class Reshape1DAvgPool;
class Reshape1DMaxPool;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::Reshape1DConvolution : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Reshape1DConvolution", "0");
    Reshape1DConvolution();
};

class ngraph::pass::Reshape1DAvgPool : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Reshape1DAvgPool", "0");
    Reshape1DAvgPool();
};

class ngraph::pass::Reshape1DMaxPool : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Reshape1DMaxPool", "0");
    Reshape1DMaxPool();
};

class ngraph::pass::Reshape1DOps : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("Reshape1DOps", "0");
    Reshape1DOps() {
        add_matcher<ngraph::pass::Reshape1DConvolution>();
        add_matcher<ngraph::pass::Reshape1DAvgPool>();
        add_matcher<ngraph::pass::Reshape1DMaxPool>();
    }
};
