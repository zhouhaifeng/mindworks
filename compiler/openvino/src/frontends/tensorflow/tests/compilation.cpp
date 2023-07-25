// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exec_graph_info.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>

#include "gtest/gtest.h"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow::tests;

class CompileModelsTests : public ::testing::Test {};

TEST_F(CompileModelsTests, NgramCompilation) {
    ov::Core core;
    auto model = convert_model("model_ngram/model_ngram.pbtxt");
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    const auto runtime_model = compiled_model.get_runtime_model();

    EXPECT_EQ(runtime_model->get_ordered_ops().size(), 4);
    EXPECT_EQ(runtime_model->get_parameters().size(), 2);
    EXPECT_EQ(runtime_model->get_results().size(), 1);
}

TEST_F(CompileModelsTests, ModelWithSplitConvConcat) {
    {
        auto model = convert_model("split_conv_concat/split_conv_concat.pbtxt");
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        const auto runtime_model = compiled_model.get_runtime_model();
        auto get_layer_type = [](const std::shared_ptr<ov::Node>& node) {
            return node->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
        };
        const auto ops = runtime_model->get_ops();
        EXPECT_EQ(0, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                      return get_layer_type(node) == "Split";
                  }));
        EXPECT_EQ(2, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                      return get_layer_type(node) == "Convolution";
                  }));
        EXPECT_EQ(0, std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                      return get_layer_type(node) == "Concat";
                  }));
    }
}
