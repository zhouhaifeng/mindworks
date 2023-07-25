// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "convolution_inst.h"
#include "pass_manager.h"
#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_padding, groupconv_with_output) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout = layout{{1, 18, 76, 135}, data_types::f16, format::bfyx};
    auto weights_data = rg.generate_random_5d<FLOAT16>(1, 18, 1, 3, 3, -1, 1);
    auto weights_mem = engine.allocate_memory({ {18, 1, 1, 3, 3}, data_types::f16, format::bfzyx});
    set_values(weights_mem, weights_data);

    topology topo;
    topo.add(input_layout("input", in_layout));
    topo.add(data("weight", weights_mem));
    topo.add(convolution("conv", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {2, 2}, true));
    topo.add(reorder("reorder", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topo, config, false, true);
    reorder_factory rf;
    program_wrapper::apply_opt_pass<prepare_padding>(*prog, true);
    const auto& node = prog->get_node("reorder_input_conv");
    auto params = node.get_kernel_impl_params();
    ASSERT_EQ(params->get_output_layout().data_padding.upper_size().spatial[2], 0);
}
