// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/non_zero.hpp"

#include "intel_gpu/primitives/non_zero.hpp"

namespace ov {
namespace intel_gpu {

static void CreateNonZeroOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layer_name = layer_type_name_ID(op);

    cldnn::primitive_id count_prim_id = layer_name + "_count";
    auto count_prim = cldnn::count_nonzero(count_prim_id,
                                           inputs[0]);

    auto gather_prim = cldnn::gather_nonzero(layer_name,
                                             inputs[0],
                                             count_prim_id);

    p.add_primitive(*op, count_prim);
    p.add_primitive(*op, gather_prim);
}

REGISTER_FACTORY_IMPL(v3, NonZero);

}  // namespace intel_gpu
}  // namespace ov
