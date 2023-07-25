// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"

#include <memory>
#include <ngraph/pass/manager.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/op_conversions/convert_broadcast3.hpp"
#include "transformations/op_conversions/convert_shapeof3.hpp"
#include "transformations/op_conversions/convert_shuffle_channels3.hpp"
#include "transformations/op_conversions/convert_topk3.hpp"
#include "transformations/op_conversions/softplus_decomposition.hpp"

bool ov::pass::ConvertOpSet3ToOpSet2::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(ConvertOpSet3ToOpSet2);
    ngraph::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);

    manager.register_pass<ov::pass::ConvertBroadcast3>();
    manager.register_pass<ov::pass::ConvertShapeOf3>();
    manager.register_pass<ov::pass::ConvertShuffleChannels3>();
    manager.register_pass<ov::pass::ConvertTopK3>();
    manager.register_pass<ov::pass::SoftPlusDecomposition>();

    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
