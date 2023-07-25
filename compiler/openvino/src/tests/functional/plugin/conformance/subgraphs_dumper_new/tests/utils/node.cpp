// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "utils/node.hpp"

#include "openvino/op/ops.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

TEST(NodeUtilsTest, get_const_ranges) {
    std::vector<float> values = {-1, -2.05, -3.65, 0, 5, 7};
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 3}), values);
    auto range = get_const_ranges<float>(const_node);
    auto range_ref = InputInfo::Range(-3.65, 7);
    ASSERT_EQ(range, range_ref);
}

TEST(NodeUtilsTest, get_input_info_by_node) {
    std::vector<float> values = {-1, -2.05, -3.65, 0, 5, 7};
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 3}), values);
    const_node->set_friendly_name("const_0");
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({2, 3}));
    param->set_friendly_name("param_0");
    auto add_node = std::make_shared<ov::op::v1::Add>(param, const_node);

    std::map<std::string, InputInfo> ref_test_info = {
        { "const_0", InputInfo(-3.65, 7, true) },
        { "param_0", InputInfo() },
    };
    std::map<std::string, InputInfo> orig_test_info = get_input_info_by_node(add_node);
    ASSERT_EQ(ref_test_info, orig_test_info);
}

TEST(NodeUtilsTest, clone_node) {
    std::vector<float> values(512, 1.f);
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 256}), values);
    const_node->set_friendly_name("const_0");
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({2, 256}));
    param->set_friendly_name("param_0");
    auto add_node_0 = std::make_shared<ov::op::v1::Add>(param, const_node);
    auto erf_node_0 = std::make_shared<ov::op::v0::Erf>(add_node_0);
    auto erf_node_1 = std::make_shared<ov::op::v0::Erf>(const_node);
    auto add_node_1 = std::make_shared<ov::op::v1::Add>(erf_node_0, erf_node_1);

    {
        auto cloned_node = clone_node(add_node_1);
        ASSERT_TRUE(ov::op::util::is_parameter(cloned_node->get_input_node_shared_ptr(0)));
        ASSERT_TRUE(ov::op::util::is_parameter(cloned_node->get_input_node_ptr(1)));
    }
    {
        auto cloned_node = clone_node(add_node_1, true);
        ASSERT_TRUE(ov::op::util::is_parameter(cloned_node->get_input_node_ptr(0)));
        ASSERT_TRUE(ov::op::util::is_constant(cloned_node->get_input_node_ptr(1)));
    }
    {
        add_node_1 = std::make_shared<ov::op::v1::Add>(const_node, erf_node_1);
        auto cloned_node = clone_node(add_node_1, true, true);
        ASSERT_TRUE(ov::op::util::is_constant(cloned_node->get_input_node_ptr(0)));
        ASSERT_TRUE(ov::op::util::is_constant(cloned_node->get_input_node_ptr(1)));
    }
}

TEST(NodeUtilsTest, generate_model_by_node) {
    std::vector<float> values = {-1, -2.05, -3.65, 0, 5, 7};
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 3}), values);
    const_node->set_friendly_name("const_0");
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({2, 3}));
    param->set_friendly_name("param_0");
    auto add_node_0 = std::make_shared<ov::op::v1::Add>(param, const_node);
    auto erf_node_0 = std::make_shared<ov::op::v0::Erf>(add_node_0);
    auto erf_node_1 = std::make_shared<ov::op::v0::Erf>(const_node);
    auto add_node_1 = std::make_shared<ov::op::v1::Add>(erf_node_0, erf_node_1);

    auto model = generate_model_by_node(add_node_1);
    auto param_0 = model->inputs().begin() ->get_node_shared_ptr();
    ASSERT_TRUE(ov::op::util::is_parameter(param_0));
    ASSERT_EQ(param_0->get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(param_0->get_element_type(), ov::element::Type_t::f32);

    auto param_1 = model->inputs().begin()->get_node_shared_ptr();
    ASSERT_TRUE(ov::op::util::is_parameter(param_1));
    ASSERT_EQ(param_1->get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(param_1->get_element_type(), ov::element::Type_t::f32);

    auto res_0 = model->outputs().rbegin()->get_node_shared_ptr();
    ASSERT_TRUE(ov::op::util::is_output(res_0));
    ASSERT_EQ(res_0->get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(res_0->get_element_type(), ov::element::Type_t::f32);
}

}  // namespace
