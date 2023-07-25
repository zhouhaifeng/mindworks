// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/any_output.hpp"

#include "ngraph/pattern/matcher.hpp"

using namespace std;

bool ov::pass::pattern::op::AnyOutput::match_value(Matcher* matcher,
                                                   const Output<Node>& pattern_value,
                                                   const Output<Node>& graph_value) {
    return input_value(0).get_node()->match_node(matcher, graph_value);
}
