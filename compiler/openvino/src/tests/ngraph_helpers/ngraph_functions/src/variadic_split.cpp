// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
        std::shared_ptr<ngraph::Node> makeVariadicSplit(const ngraph::Output<Node> &in,
                                                        const std::vector<size_t> numSplits,
                                                        int64_t axis) {
            auto splitAxisOp = std::make_shared<ngraph::opset3::Constant>(element::i64, ngraph::Shape{},
                                                                          std::vector<int64_t>{axis});
            auto numSplit = std::make_shared<ngraph::opset3::Constant>(element::u64, ngraph::Shape{numSplits.size()},
                                                                       numSplits);
            auto VariadicSplitNode = std::make_shared<ngraph::opset3::VariadicSplit>(in, splitAxisOp, numSplit);
            return VariadicSplitNode;
        }
}  // namespace builder
}  // namespace ngraph
