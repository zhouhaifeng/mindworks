// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class CompressQuantizeWeights;
class ZeroPointOptimizer;

}  // namespace pass
}  // namespace ngraph

/*
    CompressQuantizeWeights transformation goal is to pre-quantize data to minimize runtime calculations with constant
   data. To achieve this goal we perform FakeQuantize decomposition to separate quantization from dequantization in it.

    Initial graph (FakeQuantize where all inputs are Constants):

                                   |  |  |  |  |
                                   |  |  |  |  |
                                   v  v  v  v  v
                                  +------------+
                                  |FakeQuantize|
                                  +------------+
                                        |
                                        v

    is replaced to:
                                +-----------------+
                                |    Constant     |
                                | (low precision) |
                                +-----------------+
                                        |
                                        v
                                +------------------+
                                |     Convert      |
                                |  (to high prec)  |
                                +------------------+
                                        |
                                        v
                  +----------+    +------------+
                  |zero point|--->|  Subtract  |
                  +----------+    +-----+------+
                                        |
                                        v
                   +---------+    +------------+
                   |  scale  |--->|  Multiply  |
                   +---------+    +-----+------+
                                        |
                                        v

    Transformation prepares quantized constant data for Low Precision pipeline.
    Such constant data packing reduces IR size (.bin file size) in offline transformations.
    With that we can skip same calculations in the runtime and make loading of such sub-graphs to the plugin faster.
*/
class ngraph::pass::CompressQuantizeWeights : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("CompressQuantizeWeights", "0");
    CompressQuantizeWeights();
};

/*
   if zero_point == 0 we can eliminate Subtract from following dequantization subgraph:

                                +-----------------+
                                |    Constant     |
                                | (low precision) |
                                +-----------------+
                                        |
                                        v
                                +------------------+
                                |     Convert      |
                                |  (to high prec)  |
                                +------------------+
                                        |
                                        v
                  +----------+    +------------+
                  |zero point|--->|  Subtract  |
                  +----------+    +-----+------+
                                        |
                                        v
*/
class ngraph::pass::ZeroPointOptimizer : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ZeroPointOptimizer");
    ZeroPointOptimizer();
};
