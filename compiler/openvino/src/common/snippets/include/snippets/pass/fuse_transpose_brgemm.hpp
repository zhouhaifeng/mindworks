// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "openvino/op/transpose.hpp"

#include "snippets/lowered/port_descriptor.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface FuseTransposeBrgemm
 * @brief Fuses Transpose with Brgemm node, fusing on both Brgemm inputs and output is supported. Applicable to
 *        Transposes that don't change the position of the last dimension (since Brgemm supports strided rows i/o),
 *        but only 0213 Transpose is currently supported.
 * @ingroup snippets
 */
class FuseTransposeBrgemm: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseTransposeBrgemm", "0");
    FuseTransposeBrgemm();
    static const std::set<std::vector<int>> supported_cases;

private:
    static bool is_supported_transpose(const Output<Node>& transpose_port);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov