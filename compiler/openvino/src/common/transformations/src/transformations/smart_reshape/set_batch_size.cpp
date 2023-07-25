// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <itt.hpp>
#include <memory>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/smart_reshape/mimic_set_batch_size.hpp>
#include <transformations/smart_reshape/reshape_to_1D.hpp>
#include <transformations/smart_reshape/set_batch_size.hpp>
#include <transformations/smart_reshape/strided_slice_squeeze.hpp>

bool ov::pass::SetBatchSize::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(SetBatchSize);

    ngraph::pass::Manager manager;
    // This pass must be called first in pipeline
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::SharedSqueeze>();
    manager.register_pass<ov::pass::SqueezeStridedSlice>();
    manager.register_pass<ov::pass::StridedSliceSqueeze>();
    manager.register_pass<ov::pass::MimicSetBatchSize>();
    manager.run_passes(f);
    return true;
}
