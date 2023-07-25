// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API UnrollTensorIterator;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Unrolls the body of the TensorIterator layer. Multiple body copies, the number of which is determined by
 * the number of iterations of the TensorIterator layer, are created and connected to each other and to the external
 * network. If the number of TensorIterator iterations is greater than 1, then additional Concat and Split layers
 * are added to the network.
 */

class ov::pass::UnrollTensorIterator : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("UnrollTensorIterator", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
