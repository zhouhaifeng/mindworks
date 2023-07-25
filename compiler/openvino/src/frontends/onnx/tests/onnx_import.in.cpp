// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "default_opset.hpp"
#include "openvino/opsets/opset12.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "onnx_import/core/null_node.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "common_test_utils/all_close.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

// ############################################################################ CORE TESTS
OPENVINO_TEST(${BACKEND_NAME}, onnx_test_test_case) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1});
    test_case.add_input<float>({2});
    test_case.add_input<float>({3});
    test_case.add_expected_output<float>(Shape{1}, {6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_test_test_case_mutliple_inputs) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_output_names_check) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/split_equal_parts_default.onnx"));

    std::size_t size = function->get_output_size();
    for (std::size_t i{0}; i < size; ++i) {
        std::shared_ptr<Node> node = function->get_output_op(i);
        EXPECT_EQ(node->get_friendly_name(), "output_" + std::to_string(i + 1) + "/sink_port_0");
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_node_names_check) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    // Filter out Add nodes from the function graph
    std::vector<std::shared_ptr<Node>> additions;
    auto ordered_ops = function->get_ordered_ops();
    std::copy_if(ordered_ops.begin(), ordered_ops.end(), std::back_inserter(additions), [](std::shared_ptr<Node> op) {
        return std::string(op->get_type_name()) == "Add";
    });

    EXPECT_EQ(additions.size(), 2);
    EXPECT_EQ(additions.at(0)->get_friendly_name(), "add_node1");
    EXPECT_EQ(additions.at(0)->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"X"});
    EXPECT_EQ(additions.at(1)->get_friendly_name(), "Y");
    EXPECT_EQ(additions.at(1)->get_output_tensor(0).get_names(), std::unordered_set<std::string>{"Y"});
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_duplicated_output_name) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/duplicated_output_name.onnx"));
    EXPECT_EQ(function->get_output_size(), 2);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_duplicated_more_output_names) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/duplicated_more_output_names.onnx"));
    EXPECT_EQ(function->get_output_size(), 4);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1, 2}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.add_expected_output(Shape{1}, std::vector<float>{7});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_binary_add_abc) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(Inputs{{1}, {2}, {3}});
    test_case.add_expected_output(Shape{1}, std::vector<float>{6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_const_op) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/bool_const_op.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output(std::vector<bool>{1, 0, 0, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_init_and) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/bool_init_and.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output(std::vector<bool>{1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_input_or) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/bool_input_or.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(std::vector<bool>{true, false, true, false});
    test_case.add_input(std::vector<bool>{false, false, true, true});
    test_case.add_expected_output(std::vector<bool>{1, 0, 1, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_bool_init_raw) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/bool_init_raw.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output(std::vector<bool>{true, false, true});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_abc_initializers) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/add_abc_initializers.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>({3, 6, 9, 12});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_override_op) {
    onnx_import::register_operator("FalseAdd", 1, "", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    onnx_import::register_operator("FalseAdd", 1, "", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Subtract>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/override_op.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{3.f, 2.f, 1.f, 0.f});

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({-3.f, -1.f, 1.f, 3.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_import_non_existing_file) {
    try {
        onnx_import::import_onnx_model(
            file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/i.dont.exist"));
    } catch (const std::runtime_error& exc) {
        // asserts that an exception was thrown and that the error message contains the file name
        std::string msg{exc.what()};
        EXPECT_TRUE(msg.find("i.dont.exist") != std::string::npos);
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsupported_op) {
    try {
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/unsupported_op.onnx"));
        FAIL() << "Expected ngraph::ngraph_error";
    } catch (ngraph::ngraph_error const& err) {
        std::string what{err.what()};
        EXPECT_NE(what.find("OpenVINO does not support"), std::string::npos);
        EXPECT_NE(what.find("FakeOpName"), std::string::npos);
        EXPECT_NE(what.find("AnotherFakeOpName"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected ngraph::ngraph_error";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_custom_op) {
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/custom_operator.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_custom_op_register_unregister) {
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/custom_operator.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();

    onnx_import::unregister_operator("AddQ", 1, "com.intel.ai");
    try {
        auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                            SERIALIZED_ZOO,
                                                                            "onnx/custom_operator.onnx"));
        FAIL() << "Expected ngraph::ngraph_error";
    } catch (ngraph::ngraph_error const& err) {
        std::string what{err.what()};
        EXPECT_NE(what.find("OpenVINO does not support the following ONNX operations:"), std::string::npos);
    } catch (...) {
        FAIL() << "Expected ngraph::ngraph_error";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_custom_op_default_domain) {
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/custom_operator_default_domain.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({3.f, 6.f, 9.f, 12.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_op_supported) {
    // Simple case
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 1, "ai.onnx"));
    // With fallback
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 100, "ai.onnx"));

    // Different opset versions
    EXPECT_TRUE(onnx_import::is_operator_supported("Add", 1, "ai.onnx"));
    EXPECT_TRUE(onnx_import::is_operator_supported("Add", 7, "ai.onnx"));

    // Default domain name
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 1));

    // Unregistered operator
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 1));
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 1, "ai.onnx"));
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 10, "ai.onnx"));

    // Operator with bad domain name
    EXPECT_FALSE(onnx_import::is_operator_supported("Sum", 1, "bad.domain"));

    // Registered custom operator
    onnx_import::register_operator("AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });
    EXPECT_TRUE(onnx_import::is_operator_supported("AddQ", 1, "com.intel.ai"));
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_missing_op_domain) {
    onnx_import::register_operator("CustomAdd", 1, "custom.op", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    EXPECT_TRUE(onnx_import::is_operator_supported("CustomAdd", 1, "custom.op"));

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/missing_op_domain.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({0.f, 2.f, 4.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_custom_op_in_supported_operators) {
    onnx_import::register_operator("CustomAdd", 1, "custom.op", [](const onnx_import::Node& node) -> OutputVector {
        OutputVector ng_inputs{node.get_ng_inputs()};
        return {std::make_shared<ngraph::op::v1::Add>(ng_inputs.at(0), ng_inputs.at(1))};
    });

    const auto& supported_ops = onnx_import::get_supported_operators(1, "custom.op");
    EXPECT_NE(std::find(std::begin(supported_ops), std::end(supported_ops), "CustomAdd"), std::end(supported_ops));
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unknown_domain) {
    // the importer should not throw when it encounters an unknown domain in the model
    EXPECT_NO_THROW(onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/unknown_domain.onnx")));
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_op_in_unknown_domain) {
    try {
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/unknown_domain_add.onnx"));

        FAIL() << "The onnx_importer did not throw for unknown domain and op";
    } catch (const ngraph::ngraph_error& e) {
        const std::string msg = e.what();

        EXPECT_NE(msg.find("unknown.domain.Add"), std::string::npos)
            << "The error message should contain domain and op name: unknown.domain.Add";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_missing_input) {
    onnx_import::register_operator("TestMissingInOut",
                                   1,
                                   "com.intel.ai",
                                   [](const onnx_import::Node& node) -> OutputVector {
                                       OutputVector ng_inputs{node.get_ng_inputs()};
                                       Output<ngraph::Node> A = ng_inputs.at(0);
                                       Output<ngraph::Node> B = ng_inputs.at(1);
                                       Output<ngraph::Node> C = ng_inputs.at(2);

                                       A = std::make_shared<op::v1::Multiply>(A, C);
                                       if (!ngraph::op::is_null(B)) {
                                           B = std::make_shared<op::v1::Divide>(B, C);
                                       }

                                       C = std::make_shared<ngraph::op::v1::Add>(C, C);
                                       return {A, B, C};
                                   });

    onnx_import::register_operator("TestMissingIn",
                                   1,
                                   "com.intel.ai",
                                   [](const onnx_import::Node& node) -> OutputVector {
                                       OutputVector ng_inputs{node.get_ng_inputs()};
                                       std::shared_ptr<ngraph::Node> result =
                                           std::make_shared<ngraph::op::Constant>(element::f32,
                                                                                  ngraph::Shape{2, 2},
                                                                                  std::vector<float>{1, 1, 1, 1});

                                       for (const auto& ng_input : ng_inputs) {
                                           if (!ngraph::op::is_null(ng_input)) {
                                               result = std::make_shared<op::v1::Multiply>(ng_input, result);
                                           }
                                       }

                                       return {result};
                                   });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/missing_input.onnx"));

    Inputs inputs{{1, 2, 3, 4}, {5, 6, 7, 8}};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({50, 144, 294, 512});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_initializer_wo_input) {
    // This test checks a model which has an initializer, but no input with the same name
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/initializer_wo_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<float>({0, 2, 6, 12, 20, 30});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_function_dependency_to_created_subgraph) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/transformations/greater_or_equal.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5}, {3.f, 5.f, 3.f, 3.f, 6.f});
    test_case.add_input<float>(Shape{5}, {1.f, 4.f, 3.f, 7.f, 8.f});
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 1, 1, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_function_greater_or_equal_inside_if) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/transformations/greater_or_equal_inside_if.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // case when condition == true and any(x >= y)
    // expected value == x * y
    std::vector<float> x(40, 2);
    std::vector<float> y(40);
    std::iota(y.begin(), y.end(), -20.f);
    std::vector<float> expected;
    std::transform(x.begin(), x.end(), y.begin(), std::back_inserter(expected), [](float i, float j) -> float {
        return i * j;
    });
    test_case.add_input<bool>({true});  // condition
    test_case.add_input<float>(x);
    test_case.add_input<float>(y);
    test_case.add_expected_output<float>(expected);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_context_dependent_function) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/transformations/softmax_crossentropy_consumed.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{3, 5},
                               {0.54881352186203f,
                                0.7151893377304077f,
                                0.6027633547782898f,
                                0.5448831915855408f,
                                0.42365479469299316f,
                                0.6458941102027893f,
                                0.4375872015953064f,
                                0.891772985458374f,
                                0.9636627435684204f,
                                0.3834415078163147f,
                                0.7917250394821167f,
                                0.5288949012756348f,
                                0.5680445432662964f,
                                0.9255966544151306f,
                                0.07103605568408966f});
    test_case.add_input<int64_t>(Shape{3}, {1, 4, 3});
    test_case.add_expected_output<int32_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_expand_function_with_initializers) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/transformations/celu_with_initializers.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>({0.5, 1.0, 1.5, 2.0});
    test_case.run();
}

// ############################################################################ OPERATOR TESTS
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_addmul_abc) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/addmul_abc.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({9, 10, 11, 12});
    test_case.add_input<float>({5, 6, 7, 8});
    test_case.add_input<float>({1, 2, 3, 4});
    test_case.add_expected_output<float>(Shape{1, 2, 2}, {46, 62, 80, 100});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_no_keepdims) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/argmin_no_keepdims.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({2, 1, 3, 10});
    test_case.add_expected_output<int64_t>(Shape{2}, {1, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_batch_norm_default) {
    // Batch Normalization with default parameters
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f});  // data {1, 2, 1, 3}
    test_case.add_input<float>({1.f, 1.5f});                      // scale
    test_case.add_input<float>({0.f, 1.f});                       // bias
    test_case.add_input<float>({0.f, 3.f});                       // mean
    test_case.add_input<float>({1.f, 1.5f});                      // var
    test_case.add_expected_output<float>(Shape{1, 2, 1, 3},
                                         {-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_relu) {
    // Simple ReLU test
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/relu.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1, -2, 0, 1, 2, 3});
    test_case.add_expected_output<float>({0, 0, 0, 1, 2, 3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum_opset1) {
    // Simple Sum test for opset1.
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sum_opset1.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 9.f, 12.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum) {
    // Simple Sum test for opset8.
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sum.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});
    test_case.add_expected_output<float>(Shape{3}, {6.f, 12.f, 13.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum_one_input) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sum_one_input.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_expected_output<float>({3.f, 0.f, 2.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_1d) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/cum_sum_1d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f});
    test_case.add_expected_output<float>(Shape{3}, {1.f, 3.f, 6.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_axis_input) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/cum_sum_2d_axis_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_dynamic_axis_input) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/cum_sum_2d_dynamic_axis_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_input<std::int32_t>({1});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_axis_input_1d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/cum_sum_2d_axis_input_1d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 3.f, 6.f, 4.f, 9.f, 15.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_2d_dynamic_axis_input_1d) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/cum_sum_2d_dynamic_axis_input_1d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_input<std::int64_t>({0});
    test_case.add_expected_output<float>(Shape{2, 3}, {1.f, 2.f, 3.f, 5.f, 7.f, 9.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cum_sum_3d_exclusive_reverse) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/cum_sum_3d_exclusive_reverse.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f, 12.f,
                                13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
    test_case.add_expected_output<float>(Shape{2, 3, 4},
                                         {13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                                          0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f,  0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs_opset1) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/min_two_inputs_opset1.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 1.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_min_two_inputs) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/min_two_inputs.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({2.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_expected_output<float>({1.f, 2.f, 2.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_opset1) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/max_opset1.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 2.f, 1.f});
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/max.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.f, 4.f, 4.f});
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({2.f, 5.f, 3.f});

    test_case.add_expected_output<float>({3.f, 5.f, 4.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mean_opset1) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mean_opset1.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f, 0.f, 2.f});
    test_case.add_input<float>({1.f, 3.f, 4.f});
    test_case.add_input<float>({2.f, 6.f, 6.f});

    test_case.add_expected_output<float>({2.f, 3.f, 4.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mean) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mean.onnx"));

    // input data shape (3, )
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.f});
    test_case.add_input<float>({1.f, 2.f, 5.f});
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gemm_abc) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/gemm_abc.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{19, 20, 21, 22},
                                                 {23, 24, 25, 26},
                                                 {27, 28, 29, 30},
                                                 {31, 32, 33, 34},
                                                 {35, 36, 37, 38},
                                                 {39, 40, 41, 42}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}).get_vector());

    auto expected_output =
        test::NDArray<float, 2>({{340, 350.5, 361, 371.5}, {862, 890.5, 919, 947.5}, {1384, 1430.5, 1477, 1523.5}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_matmul) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/matmul.onnx"));

    std::vector<std::vector<float>> inputs;

    inputs.emplace_back(test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}).get_vector());

    auto expected_output = test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_0D) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softmax_0D.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>({1.0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_1D) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softmax_1D.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.0, 0.0, 1.0});
    test_case.add_expected_output<float>({0.09003058f, 0.24472848f, 0.66524094f});
    test_case.run();
}
namespace {
// common input for all Softmax 3D test cases (Shape = {3,4,5})
// clang-format off
const std::vector<float> SOFTMAX_INPUT = {
    2.75793882f,  -0.50841322f, 0.82013929f,  -0.62409912f, -0.96136118f,
    0.21004745f,  1.38337255f,  1.19030397f,  2.0940445f,   -0.03551657f,
    -0.78686039f, 1.992782f,    0.04300319f,  -0.29230777f, -0.56797112f,
    -1.26732165f, -0.61935399f, 0.57670432f,  0.92844898f,  2.82469233f,

    0.98721677f,  -0.05100663f, -1.21178917f, -0.17530157f, 1.40051805f,
    -0.13259761f, -1.14313018f, 0.2673723f,   -0.87996154f, 1.29053106f,
    1.55f,        0.8396538f,   1.20729817f,  0.23727845f,  -0.89113606f,
    -1.70909842f, 0.26460363f,  -0.70566808f, 2.383518f,    1.07024615f,

    -1.21722605f, 0.82919357f,  0.55765697f,  0.12657686f,  0.63432172f,
    0.75425957f,  -2.43721014f, -1.24478184f, 2.65316853f,  1.19509542f,
    -0.95523998f, 0.5149006f,   -0.01151649f, 0.68327026f,  -0.4589638f,
    -0.46554745f, 0.21055324f,  0.39266729f,  2.05098086f,  1.83207919f};
}  // namespace
// clang-format on

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_0) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softmax_axis_0.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.09683057f, 0.00369363f, 0.01394559f, 0.00329012f, 0.00234823f,
         0.00757665f, 0.02449322f, 0.02019284f, 0.04985249f, 0.00592694f,
         0.00279593f, 0.04505148f, 0.00641108f, 0.00458466f, 0.00348007f,
         0.00172928f, 0.00330577f, 0.01093237f, 0.01554086f, 0.10351497f,

         0.01648154f, 0.00583583f, 0.00182802f, 0.00515374f, 0.02491679f,
         0.00537859f, 0.00195794f, 0.00802367f, 0.00254737f, 0.0223216f,
         0.02893419f, 0.0142204f,  0.02053893f, 0.00778581f, 0.00251907f,
         0.00111174f, 0.00800149f, 0.0030324f,  0.06658917f, 0.0179084f,

         0.00181811f, 0.01407243f, 0.01072611f, 0.0069699f,  0.01158077f,
         0.01305647f, 0.00053677f, 0.0017687f,  0.08719896f, 0.02028982f,
         0.00236265f, 0.01027717f, 0.0060709f,  0.01216173f, 0.00388087f,
         0.00385541f, 0.00758048f, 0.00909469f, 0.04775123f, 0.03836337f});
    // clang-format on

    test_case.run(6);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_1) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softmax_axis_1.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.22757064f, 0.00868076f, 0.03277484f, 0.00773243f, 0.0055188f,
         0.0178066f,  0.05756383f, 0.04745709f, 0.11716303f, 0.01392945f,
         0.00657097f, 0.10587974f, 0.01506727f, 0.01077484f, 0.00817884f,
         0.00406413f, 0.00776921f, 0.0256932f,  0.03652405f, 0.24328028f,

         0.06217413f, 0.02201481f, 0.00689594f, 0.01944171f, 0.09399488f,
         0.02028993f, 0.00738604f, 0.03026811f, 0.00960958f, 0.08420492f,
         0.10914991f, 0.05364435f, 0.07748005f, 0.02937079f, 0.0095028f,
         0.00419387f, 0.03018442f, 0.01143929f, 0.2511977f,  0.06755678f,

         0.00587593f, 0.04548053f, 0.0346656f,  0.02252594f, 0.03742775f,
         0.04219705f, 0.00173478f, 0.00571623f, 0.2818174f,  0.06557446f,
         0.00763582f, 0.03321466f, 0.01962049f, 0.03930537f, 0.01254255f,
         0.01246025f, 0.02449929f, 0.02939305f, 0.15432668f, 0.12398617f});
    // clang-format on

    test_case.run(4);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_1_opset11) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/softmax_axis_1_opset11.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.88890495f, 0.04825497f, 0.27088348f, 0.04490523f, 0.02037154f,
         0.06955369f, 0.31998834f, 0.39223197f, 0.68041159f, 0.05141776f,
         0.02566661f, 0.5885689f,  0.12453075f, 0.06257374f, 0.03019055f,
         0.01587475f, 0.0431878f,  0.21235381f, 0.21210944f, 0.89802015f,

         0.31752626f, 0.19442629f, 0.0546935f,  0.06279221f, 0.36823282f,
         0.10362164f, 0.06523066f, 0.24006419f, 0.03103672f, 0.32987983f,
         0.55743381f, 0.473766f,   0.61451431f, 0.09486084f, 0.03722801f,
         0.02141829f, 0.26657706f, 0.090728f,   0.81131024f, 0.26465935f,

         0.08619648f, 0.43343993f, 0.3877785f,  0.04523505f, 0.15625437f,
         0.61900597f, 0.01653285f, 0.06394322f, 0.56592636f, 0.27376196f,
         0.11201305f, 0.31654337f, 0.21947994f, 0.07893034f, 0.05236297f,
         0.18278451f, 0.23348385f, 0.32879834f, 0.30990825f, 0.5176207f});
    // clang-format on

    test_case.run(4);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_negative_1_opset11) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/softmax_axis_negative_1_opset11.onnx"));

    auto test_case = test::TestCase(function);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.80619484f, 0.03075256f, 0.1161086f,  0.027393f,   0.01955098f,
         0.07012683f, 0.22670066f, 0.18689778f, 0.4614171f,  0.05485764f,
         0.04486171f, 0.7228683f,  0.10286818f, 0.07356264f, 0.05583908f,
         0.01280724f, 0.02448298f, 0.08096659f, 0.11509769f, 0.76664555f,

         0.30399805f, 0.10764059f, 0.03371745f, 0.09505949f, 0.4595844f,
         0.13369875f, 0.04866969f, 0.19944906f, 0.0633215f,  0.554861f,
         0.39101103f, 0.19217177f, 0.27755913f, 0.10521588f, 0.03404216f,
         0.01150354f, 0.08279411f, 0.03137731f, 0.6890207f,  0.18530433f,

         0.0402528f,  0.31156224f, 0.23747502f, 0.15431291f, 0.25639707f,
         0.10627912f, 0.00436928f, 0.01439711f, 0.7097961f,  0.16515835f,
         0.06798343f, 0.29571748f, 0.17468554f, 0.34994435f, 0.11166911f,
         0.03615172f, 0.07108136f, 0.08527993f, 0.4477579f,  0.35972902f});
    // clang-format on

    test_case.run(6);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softmax_axis_negative_1_opset13) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/softmax_axis_negative_1_opset13.onnx"));

    auto test_case = test::TestCase(function);
    test_case.add_input<float>(SOFTMAX_INPUT);

    // clang-format off
    test_case.add_expected_output<float>(
        Shape{3, 4, 5},
        {0.80619484f, 0.03075256f, 0.1161086f,  0.027393f,   0.01955098f,
         0.07012683f, 0.22670066f, 0.18689778f, 0.4614171f,  0.05485764f,
         0.04486171f, 0.7228683f,  0.10286818f, 0.07356264f, 0.05583908f,
         0.01280724f, 0.02448298f, 0.08096659f, 0.11509769f, 0.76664555f,

         0.30399805f, 0.10764059f, 0.03371745f, 0.09505949f, 0.4595844f,
         0.13369875f, 0.04866969f, 0.19944906f, 0.0633215f,  0.554861f,
         0.39101103f, 0.19217177f, 0.27755913f, 0.10521588f, 0.03404216f,
         0.01150354f, 0.08279411f, 0.03137731f, 0.6890207f,  0.18530433f,

         0.0402528f,  0.31156224f, 0.23747502f, 0.15431291f, 0.25639707f,
         0.10627912f, 0.00436928f, 0.01439711f, 0.7097961f,  0.16515835f,
         0.06798343f, 0.29571748f, 0.17468554f, 0.34994435f, 0.11166911f,
         0.03615172f, 0.07108136f, 0.08527993f, 0.4477579f,  0.35972902f});
    // clang-format on

    test_case.run(6);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sub.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/div.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_bcast) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/add_bcast.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 1>({1, 2, 3, 4, 5}).get_vector());

    auto expected_output =
        test::NDArray<float, 4>({{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
                                  {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
                                  {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_center_point_box_format) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/nonmaxsuppression_center_point_box_format.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input(
        std::vector<float>({0.5f, 0.5f,  1.0f, 1.0f, 0.5f, 0.6f,  1.0f, 1.0f, 0.5f, 0.4f,   1.0f, 1.0f,
                            0.5f, 10.5f, 1.0f, 1.0f, 0.5f, 10.6f, 1.0f, 1.0f, 0.5f, 100.5f, 1.0f, 1.0f}));  // boxes
    test_case.add_input(std::vector<float>({0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}));                        // scores
    test_case.add_input(std::vector<int64_t>({3}));   // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));  // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));  // score_threshold

    test_case.add_expected_output<int64_t>(Shape{3, 3}, {0, 0, 3, 0, 0, 0, 0, 0, 5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_single_box) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/nonmaxsuppression_single_box.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input(std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f}));  // boxes
    test_case.add_input(std::vector<float>({0.9f}));                    // scores
    test_case.add_input(std::vector<int64_t>({3}));                     // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));                    // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));                    // score_threshold

    test_case.add_expected_output<int64_t>(Shape{1, 3}, {0, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_v9_single_box) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/nonmaxsuppression_v9_single_box.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input(std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f}));  // boxes
    test_case.add_input(std::vector<float>({0.9f}));                    // scores
    test_case.add_input(std::vector<int64_t>({3}));                     // max_output_boxes_per_class
    test_case.add_input(std::vector<float>({0.5f}));                    // iou_threshold
    test_case.add_input(std::vector<float>({0.0f}));                    // score_threshold

    test_case.add_expected_output<int64_t>(Shape{1, 3}, {0, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_nonmaxsuppression_default_score_threshold) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/nms_default_score_threshold.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input(
        Shape{1, 50, 4},
        std::vector<float>(
            {278.862060546875f,   453.5412902832031f,  295.09234619140625f, 470.2095031738281f,  225.9730682373047f,
             387.33990478515625f, 241.69297790527344f, 403.43377685546875f, 281.3062438964844f,  453.8412170410156f,
             298.6865539550781f,  470.9977111816406f,  216.9517364501953f,  450.6717529296875f,  232.95777893066406f,
             466.14276123046875f, 217.54473876953125f, 449.9130859375f,     233.97265625f,       466.1539306640625f,
             279.0079650878906f,  453.865234375f,      294.8210144042969f,  470.123046875f,      226.5626983642578f,
             388.5235290527344f,  242.2290496826172f,  404.2589416503906f,  216.49752807617188f, 450.7710876464844f,
             233.07443237304688f, 466.7010192871094f,  281.3638000488281f,  454.33892822265625f, 298.5252990722656f,
             471.1678466796875f,  217.3330841064453f,  451.484130859375f,   234.1898651123047f,  466.83148193359375f,
             187.2439727783203f,  466.8524475097656f,  208.7089385986328f,  489.7967224121094f,  257.8833923339844f,
             515.705322265625f,   280.8927917480469f,  539.775146484375f,   226.52525329589844f, 387.7011413574219f,
             241.6272430419922f,  403.7854919433594f,  187.38221740722656f, 466.5717468261719f,  209.05845642089844f,
             489.4494323730469f,  217.56448364257812f, 451.1393737792969f,  233.90216064453125f, 466.1475524902344f,
             279.45611572265625f, 454.00299072265625f, 296.16424560546875f, 471.84521484375f,    279.04486083984375f,
             453.9889221191406f,  295.2816162109375f,  470.4144592285156f,  187.18997192382812f, 466.4650573730469f,
             209.26266479492188f, 488.8149719238281f,  189.04197692871094f, 469.8923034667969f,  208.8195037841797f,
             491.5357971191406f,  216.47879028320312f, 450.1073303222656f,  233.21575927734375f, 466.9475402832031f,
             278.86163330078125f, 454.966552734375f,   296.38958740234375f, 471.9764404296875f,  259.4800720214844f,
             515.1390991210938f,  282.3655090332031f,  539.4806518554688f,  285.031494140625f,   389.0125427246094f,
             302.09747314453125f, 406.9799499511719f,  285.1270446777344f,  389.06890869140625f, 301.2108459472656f,
             405.7711181640625f,  188.17117309570312f, 467.71533203125f,    208.49929809570312f, 490.401611328125f,
             278.93292236328125f, 453.8080139160156f,  295.4295654296875f,  469.9015808105469f,  279.0393371582031f,
             454.2393798828125f,  296.3529357910156f,  471.6363525390625f,  187.29873657226562f, 467.9837951660156f,
             208.29107666015625f, 489.8014221191406f,  187.79478454589844f, 466.6510314941406f,  208.3644561767578f,
             490.2976989746094f,  188.4196014404297f,  468.3448486328125f,  209.06849670410156f, 491.94384765625f,
             281.4726867675781f,  454.0541687011719f,  298.2876892089844f,  470.2845764160156f,  225.8560333251953f,
             387.4819030761719f,  241.4767608642578f,  403.4317321777344f,  280.7021484375f,     455.43206787109375f,
             297.9931640625f,     471.99749755859375f, 226.0373077392578f,  387.4749450683594f,  241.48097229003906f,
             403.4716491699219f,  259.018310546875f,   515.3871459960938f,  281.7872314453125f,  540.0093383789062f,
             217.71246337890625f, 450.4556884765625f,  234.254150390625f,   467.68182373046875f, 257.5479736328125f,
             518.8912353515625f,  280.48260498046875f, 541.3863525390625f,  216.87359619140625f, 450.3395080566406f,
             232.39752197265625f, 465.5039367675781f,  258.2445068359375f,  515.2009887695312f,  280.29803466796875f,
             540.3602905273438f,  217.54478454589844f, 451.3944091796875f,  233.6602020263672f,  467.51971435546875f,
             258.30133056640625f, 515.2357788085938f,  280.1400146484375f,  541.3275756835938f,  217.05136108398438f,
             451.8975524902344f,  232.9573974609375f,  466.9907531738281f,  215.86386108398438f, 450.801025390625f,
             232.117919921875f,   466.3701171875f,     279.01593017578125f, 453.6647644042969f,  296.13372802734375f,
             471.4644470214844f,  280.1851806640625f,  454.41900634765625f, 296.481201171875f,   471.63104248046875f,
             259.1214904785156f,  516.8644409179688f,  281.7276306152344f,  541.0162963867188f,  285.2935485839844f,
             389.03515625f,       302.1134948730469f,  406.89373779296875f, 279.6715393066406f,  455.1846923828125f,
             296.6995544433594f,  471.5782470703125f,  258.1405029296875f,  518.9312744140625f,  281.019287109375f,
             541.5760498046875f,  187.80953979492188f, 466.8480224609375f,  208.54336547851562f, 489.9696044921875f}));
    test_case.add_input(
        Shape{1, 1, 50},
        std::vector<float>(
            {5.485373497009277f,  5.469169616699219f,  5.450349807739258f,  5.446445465087891f, 5.43833065032959f,
             5.407294273376465f,  5.3790669441223145f, 5.3575520515441895f, 5.348986625671387f, 5.309826850891113f,
             5.266261577606201f,  5.230800151824951f,  5.079848766326904f,  5.066829204559326f, 4.913329601287842f,
             4.895563125610352f,  4.8786115646362305f, 4.872953414916992f,  4.825906753540039f, 4.812736511230469f,
             4.761179447174072f,  4.657320022583008f,  4.640903949737549f,  4.63286828994751f,  4.600266933441162f,
             4.599870204925537f,  4.5536088943481445f, 4.521742820739746f,  4.465426445007324f, 4.4556074142456055f,
             4.451722621917725f,  4.416017055511475f,  4.410635471343994f,  4.403003215789795f, 4.387508392333984f,
             4.3634934425354f,    4.362300872802734f,  4.348748683929443f,  4.345107555389404f, 4.32416296005249f,
             4.3132781982421875f, 4.287333965301514f,  4.223401069641113f,  4.220005035400391f, 4.179988861083984f,
             4.099865436553955f,  4.097578048706055f,  4.075544357299805f,  4.0459885597229f}));

    test_case.add_expected_output<int64_t>(Shape{7, 3},
                                           {0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 10, 0, 0, 11, 0, 0, 22});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_log_sum.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{2.77258872f}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_log_sum_exp) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/reduce_log_sum_exp.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{3.77258872f}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l1) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_l1.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_l2) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_l2.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{4}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_max) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_max.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_max_invalid_axes) {
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                     SERIALIZED_ZOO,
                                                                     "onnx/reduce_max_invalid_axes.onnx")),
                 ngraph::ngraph_error);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_mean) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_mean.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(Shape{}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_min) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_min.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_prod) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_prod.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{1}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_sum.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_dynamic_rank_input) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/reduce_sum_dynamic_rank_input.onnx"));
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_square) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reduce_sum_square.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}).get_vector()};

    // output data shape (1,)
    auto expected_output = test::NDArray<float, 4>({{{{16}}}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/reduce_sum_13_axes_as_constant.onnx"));

    Inputs inputs{test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_single_axis) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reduce_sum_13_axes_as_constant_single_axis.onnx"));

    Inputs inputs{test::NDArray<float, 3>({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}).get_vector()};

    auto test_case = test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{2, 1, 3}, {5.0f, 7.0f, 9.0f, 17.0f, 19.0f, 21.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_constant_keepdims_off) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reduce_sum_13_axes_as_constant_keepdims_off.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{test::NDArray<float, 4>({{{{1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 1.0f, 1.0f}}}})
                      .get_vector()};

    auto test_case = test::TestCase(function, s_device);

    test_case.add_expected_output<float>(Shape{}, {16.0f});

    test_case.add_multiple_inputs(inputs);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_input) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/reduce_sum_13_axes_as_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_input<int64_t>({1});

    test_case.add_expected_output<float>(Shape{2, 1}, {3.0f, 7.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_as_0_dim_input) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/reduce_sum_13_axes_as_0_dim_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    test_case.add_expected_output<float>(Shape{3, 2, 2},
                                         {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_input_dynamic) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/reduce_sum_13_input_dynamic.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    test_case.add_expected_output<int64_t>(Shape{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/reduce_sum_13_axes_empty.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_dynamic_rank_input) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reduce_sum_13_axes_empty_dynamic_rank_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_with_noop) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reduce_sum_13_axes_empty_with_noop.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(
        Shape{1, 1, 4, 4},
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reduce_sum_13_axes_empty_without_noop) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reduce_sum_13_axes_empty_without_noop.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 1, 1}, {16.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_asymertic_last_dim) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/resize10_asymertic_last_dim.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(Shape{1, 1, 1, 19},
                                         {1.0f,
                                          1.0f,
                                          2.0f,
                                          2.0f,
                                          3.0f,
                                          3.0f,
                                          4.0f,
                                          4.0f,
                                          5.0f,
                                          5.0f,
                                          6.0f,
                                          6.0f,
                                          7.0f,
                                          7.0f,
                                          8.0f,
                                          8.0f,
                                          9.0f,
                                          9.0f,
                                          10.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_asymertic_dim_in_the_middle) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize10_asymertic_dim_in_the_middle.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(Shape{1, 1, 19, 1},
                                         {1.0f,
                                          1.0f,
                                          2.0f,
                                          2.0f,
                                          3.0f,
                                          3.0f,
                                          4.0f,
                                          4.0f,
                                          5.0f,
                                          5.0f,
                                          6.0f,
                                          6.0f,
                                          7.0f,
                                          7.0f,
                                          8.0f,
                                          8.0f,
                                          9.0f,
                                          9.0f,
                                          10.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_empty_constant_as_input) {
    // this model contains a Constant node with an empty underlying tensor
    // this node is connected to the "roi" input of the Resize op but this input should be
    // ignored since the Resize coordinate_transformation_mode is set to asymmetric
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_empty_constant_as_input.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 8},
        {1.0f, 1.5f, 2.0f, 2.5f,  3.0f,  3.0f,  3.0f,  3.0f,  2.5f, 3.25f, 4.0f, 4.75f, 5.5f,  5.5f,  5.5f,  5.5f,
         4.0f, 5.0f, 6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,  4.0f, 5.0f,  6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,

         6.0f, 5.0f, 4.0f, 3.0f,  2.0f,  2.0f,  2.0f,  2.0f,  6.5f, 6.5f,  6.5f, 6.5f,  6.5f,  6.5f,  6.5f,  6.5f,
         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f, 7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_down_scales_const_linear) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize10_down_scales_const_linear.onnx"));

    // Input data shape (1, 1, 2, 4)
    // Input const scales values {1.0, 1.0, 0.6, 0.6}
    // mode: linear

    Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 2.6666665f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_down_scales_const_nearest) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize10_down_scales_const_nearest.onnx"));

    // Input data shape (1, 1, 2, 4)
    // Input const scales values {1.0, 1.0, 0.6, 0.6}
    // mode: nearest

    Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    test_case.add_expected_output<float>(expected_output_shape, {1.0, 3.0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_linear) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize10_up_scales_const_linear.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: nearest

    Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 3.0, 3.5, 4.0, 4.0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize10_up_scales_const_nearest) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize10_up_scales_const_nearest.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: linear

    Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0, 2.0, 3.0, 4.0});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                          3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_scales_linear_asymmetric) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_down_scales_linear_asymmetric.onnx"));

    const Shape expected_output_shape{1, 1, 1, 2};
    auto test_case = test::TestCase(function, s_device);
    const size_t input_size = 8;
    std::vector<float> input_data(input_size);
    std::iota(std::begin(input_data), std::end(input_data), 1.0f);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 2.66666651f});

    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor_dynamic_sizes) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/resize11_scales_nearest_asymmetric_floor_dynamic_scales.onnx"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});  // roi
    test_case.add_input<float>(std::vector<float>{1.0f, 1.0f, 2.0f, 0.5f});                          // scales
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_linear_asymmetric) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_scales_linear_asymmetric.onnx"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.5f, 2.0f, 2.5f,  3.0f,  3.0f,  3.0f,  3.0f,  2.5f, 3.25f, 4.0f, 4.75f, 5.5f,  5.5f,  5.5f,  5.5f,
         4.0f, 5.0f, 6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,  4.0f, 5.0f,  6.0f, 7.0f,  8.0f,  8.0f,  8.0f,  8.0f,

         6.0f, 5.0f, 4.0f, 3.0f,  2.0f,  2.0f,  2.0f,  2.0f,  6.5f, 6.5f,  6.5f, 6.5f,  6.5f,  6.5f,  6.5f,  6.5f,
         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f, 7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_scales_nearest_asymmetric_floor) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_scales_nearest_asymmetric_floor.onnx"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_cubic_align_corners) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_scales_cubic_align_corners.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {
            1.0f,         1.34110787f,  1.80029155f,  2.32944606f,  2.67055394f,  3.19970845f,  3.65889213f,
            4.0f,         2.36443149f,  2.70553936f,  3.16472303f,  3.69387755f,  4.03498542f,  4.56413994f,
            5.02332362f,  5.36443149f,  4.20116618f,  4.54227405f,  5.00145773f,  5.53061224f,  5.87172012f,
            6.40087464f,  6.86005831f,  7.20116618f,  6.31778426f,  6.65889213f,  7.1180758f,   7.64723032f,
            7.98833819f,  8.51749271f,  8.97667638f,  9.31778426f,  7.68221574f,  8.02332362f,  8.48250729f,
            9.01166181f,  9.35276968f,  9.8819242f,   10.34110787f, 10.68221574f, 9.79883382f,  10.13994169f,
            10.59912536f, 11.12827988f, 11.46938776f, 11.99854227f, 12.45772595f, 12.79883382f, 11.63556851f,
            11.97667638f, 12.43586006f, 12.96501458f, 13.30612245f, 13.83527697f, 14.29446064f, 14.6355685f,
            13.0f,        13.34110787f, 13.80029155f, 14.32944606f, 14.67055394f, 15.19970845f, 15.65889213f,
            16.0f,
        });
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_scales_tf_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_scales_tf_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.95703f, 2.43359f, 3.0625f,  3.46875f, 4.09766f, 4.57422f, 4.87109f, 4.80078f, 3.86328f, 4.33984f, 4.96875f,
         5.375f,   6.00391f, 6.48047f, 6.77734f, 6.70703f, 6.37891f, 6.85547f, 7.48438f, 7.89063f, 8.51953f, 8.99609f,
         9.29297f, 9.22266f, 8.00391f, 8.48047f, 9.10938f, 9.51563f, 10.1445f, 10.6211f, 10.918f,  10.8477f, 10.5195f,
         10.9961f, 11.625f,  12.0313f, 12.6602f, 13.1367f, 13.4336f, 13.3633f, 12.4258f, 12.9023f, 13.5313f, 13.9375f,
         14.5664f, 15.043f,  15.3398f, 15.2695f, 13.6133f, 14.0898f, 14.7188f, 15.125f,  15.7539f, 16.2305f, 16.5273f,
         16.457f,  13.332f,  13.8086f, 14.4375f, 14.8438f, 15.4727f, 15.9492f, 16.2461f, 16.1758f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_all_attributes_default) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_sizes_all_attributes_default.onnx"));

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
         2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f,
         3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_sizes_nearest_asymmetric_floor) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_sizes_nearest_asymmetric_floor.onnx"));

    const Shape expected_output_shape{2, 1, 4, 1};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 3.0f, 4.0f, 8.0f, 6.0f, 2.0f, 7.0f, 11.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.0f, 1.0f, 4.0f, 4.0f, 6.0f, 6.0f, 7.0f, 7.0f});

    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_linear_asymmetric) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_sizes_linear_asymmetric.onnx"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{2.0f, 4.0f, 1.0f, 3.0f, 7.0f, 8.0f, 9.0f, 6.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {2.0f, 2.5f,  3.0f, 3.5f,  4.0f, 4.0f, 4.0f, 4.0f, 1.5f, 2.0f,  2.5f, 3.0f,  3.5f, 3.5f, 3.5f, 3.5f,
         1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f,
         7.0f, 7.25f, 7.5f, 7.75f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 7.75f, 7.5f, 7.25f, 7.0f, 7.0f, 7.0f, 7.0f,
         9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f, 9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_cubic_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_down_sizes_cubic_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 3, 3};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.6307871f, 3.0046299f, 4.3784733f, 7.1261587f, 8.5f, 9.873844f, 12.621532f, 13.995373f, 15.369216f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_linear_pytorch_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_down_sizes_linear_pytorch_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 3, 1};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {1.666666f, 7.0f, 12.333333f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_sizes_cubic_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922f,  2.22332922f,  2.70807922f,  3.15907922f,
         3.49007922f,  3.67557922f,  1.39437963f,  1.57987963f,  1.91087963f,  2.36187963f,  2.84662963f,  3.16262963f,
         3.64737963f,  4.09837963f,  4.42937963f,  4.61487963f,  2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693f,  4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693f,  5.20525069f,  5.39075069f,
         5.72175069f,  6.17275069f,  6.65750069f,  6.97350069f,  7.45825069f,  7.90925069f,  8.24025069f,  8.42575069f,
         6.88975f,     7.07525f,     7.40625f,     7.85725f,     8.342f,       8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,  10.02649931f, 10.34249931f,
         10.82724931f, 11.27824931f, 11.60924931f, 11.79474931f, 10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f,
         12.28044307f, 12.59644307f, 13.08119307f, 13.53219307f, 13.86319307f, 14.04869307f, 12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037f, 14.15337037f, 14.63812037f, 15.08912037f, 15.42012037f, 15.60562037f,
         13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f, 14.77667078f, 15.09267078f, 15.57742078f, 16.02842078f,
         16.35942078f, 16.54492078f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_cubic_half_pixel_dynamic_sizes) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/resize11_up_sizes_cubic_half_pixel_dynamic_sizes.onnx"));

    const Shape expected_output_shape{1, 1, 9, 10};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_input<float>(std::vector<float>{1, 1, 9, 10});  // sizes
    test_case.add_expected_output<float>(
        expected_output_shape,
        {0.45507922f,  0.64057922f,  0.97157922f,  1.42257922f,  1.90732922f,  2.22332922f,  2.70807922f,  3.15907922f,
         3.49007922f,  3.67557922f,  1.39437963f,  1.57987963f,  1.91087963f,  2.36187963f,  2.84662963f,  3.16262963f,
         3.64737963f,  4.09837963f,  4.42937963f,  4.61487963f,  2.95130693f,  3.13680693f,  3.46780693f,  3.91880693f,
         4.40355693f,  4.71955693f,  5.20430693f,  5.65530693f,  5.98630693f,  6.17180693f,  5.20525069f,  5.39075069f,
         5.72175069f,  6.17275069f,  6.65750069f,  6.97350069f,  7.45825069f,  7.90925069f,  8.24025069f,  8.42575069f,
         6.88975f,     7.07525f,     7.40625f,     7.85725f,     8.342f,       8.658f,       9.14275f,     9.59375f,
         9.92475f,     10.11025f,    8.57424931f,  8.75974931f,  9.09074931f,  9.54174931f,  10.02649931f, 10.34249931f,
         10.82724931f, 11.27824931f, 11.60924931f, 11.79474931f, 10.82819307f, 11.01369307f, 11.34469307f, 11.79569307f,
         12.28044307f, 12.59644307f, 13.08119307f, 13.53219307f, 13.86319307f, 14.04869307f, 12.38512037f, 12.57062037f,
         12.90162037f, 13.35262037f, 13.83737037f, 14.15337037f, 14.63812037f, 15.08912037f, 15.42012037f, 15.60562037f,
         13.32442078f, 13.50992078f, 13.84092078f, 14.29192078f, 14.77667078f, 15.09267078f, 15.57742078f, 16.02842078f,
         16.35942078f, 16.54492078f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_round_prefer_floor_half_pixel) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/resize11_up_sizes_nearest_round_prefer_floor_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 7, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
         2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f,
         3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f});
    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_prefer_ceil_asymmetric) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/resize11_up_sizes_nearest_prefer_ceil_asymmetric.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {
            1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  4.0f,  4.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,
            8.0f,  8.0f,  8.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  8.0f,  8.0f,  9.0f,  10.0f,
            10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f,
            12.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f,
            15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f,
        });
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_ceil_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_sizes_nearest_ceil_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  4.0f,  4.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,
         8.0f,  8.0f,  8.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  8.0f,  8.0f,  9.0f,  10.0f,
         10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f,
         12.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f,
         15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_up_sizes_nearest_floor_align_corners) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_up_sizes_nearest_floor_align_corners.onnx"));

    const Shape expected_output_shape{1, 1, 8, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.0f, 1.0f, 2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  1.0f,  1.0f,  1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,
         1.0f, 1.0f, 1.0f, 2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  5.0f,  5.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,
         5.0f, 5.0f, 5.0f, 6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  9.0f,  9.0f,  9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f,
         9.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 13.0f, 13.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_resize11_down_sizes_tf_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/resize11_down_sizes_tf_half_pixel.onnx"));

    const Shape expected_output_shape{1, 1, 3, 2};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.0f,
                                  2.0f,
                                  3.0f,
                                  4.0f,
                                  5.0f,
                                  6.0f,
                                  7.0f,
                                  8.0f,
                                  9.0f,
                                  10.0f,
                                  11.0f,
                                  12.0f,
                                  13.0f,
                                  14.0f,
                                  15.0f,
                                  16.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f});
    test_case.run_with_tolerance_as_fp(2.0e-2f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shape) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/shape.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                                 {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<int64_t>({3, 4, 5});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_elu) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/elu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-1.999753180391830f, -1.999329074744190f, -1.998176236068890f, -1.995042495646670f, -1.986524106001830f},
              {-1.963368722222530f, -1.900425863264270f, -1.729329433526770f, -1.264241117657120f, 0},
              {1, 2, 3, 4, 5},
              {6, 7, 8, 9, 10}},
             {{-1.963368722222530f, -1.900425863264270f, -1.729329433526770f, -1.264241117657120f, 0},
              {1, 2, 3, 4, 5},
              {6, 7, 8, 9, 10},
              {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1},
              {-1.264241117657120f, -1.264241117657120f, -1.264241117657120f, -1.264241117657120f, -1.264241117657120f},
              {0, 0, 0, 0, 0},
              {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_leaky_relu) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/leaky_relu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-0.9f, -0.8f, -0.7f, -0.6f, -0.5f}, {-0.4f, -0.3f, -0.2f, -0.1f, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-0.4f, -0.3f, -0.2f, -0.1f, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-0.1f, -0.1f, -0.1f, -0.1f, -0.1f}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_nd) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/prelu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}},
                                                 {{0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}},
                                                 {{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}}})
                            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>({{{-9, 0, -7, 0, -5}, {0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {0, -1, 0, -1, 0}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_batch_nd_elementwise) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/prelu_batch_nd.onnx"));

    Inputs inputs;
    // Shape{2, 3, 4, 5}
    inputs.emplace_back(std::vector<float>{
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.});

    // Shape{2, 3, 4, 5}
    std::vector<float> slope(shape_size(Shape{2, 3, 4, 5}));
    std::iota(std::begin(slope), std::end(slope), 0.f);
    inputs.emplace_back(slope);

    // Shape{2, 3, 4, 5}
    auto expected_output = std::vector<float>{
        -0.,   -1.,   -2.,   -3.,   -4.,   -5.,   -6.,   -7.,   -8.,   -9.,   -10.,  -11.,  -12.,  -13.,  -14.,
        -15.,  -16.,  -17.,  -18.,  -19.,  -20.,  -21.,  -22.,  -23.,  -24.,  -25.,  -26.,  -27.,  -28.,  -29.,
        -30.,  -31.,  -32.,  -33.,  -34.,  -35.,  -36.,  -37.,  -38.,  -39.,  -40.,  -41.,  -42.,  -43.,  -44.,
        -45.,  -46.,  -47.,  -48.,  -49.,  -50.,  -51.,  -52.,  -53.,  -54.,  -55.,  -56.,  -57.,  -58.,  -59.,
        -60.,  -61.,  -62.,  -63.,  -64.,  -65.,  -66.,  -67.,  -68.,  -69.,  -70.,  -71.,  -72.,  -73.,  -74.,
        -75.,  -76.,  -77.,  -78.,  -79.,  -80.,  -81.,  -82.,  -83.,  -84.,  -85.,  -86.,  -87.,  -88.,  -89.,
        -90.,  -91.,  -92.,  -93.,  -94.,  -95.,  -96.,  -97.,  -98.,  -99.,  -100., -101., -102., -103., -104.,
        -105., -106., -107., -108., -109., -110., -111., -112., -113., -114., -115., -116., -117., -118., -119.};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_1d) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/prelu_1d.onnx"));

    Inputs inputs;
    // Shape{2, 3, 4, 5}
    inputs.emplace_back(std::vector<float>{
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.});

    // Shape{5}
    inputs.emplace_back(std::vector<float>{0, 1, 2, 3, 4});

    // Shape{2, 3, 4, 5}
    auto expected_output = std::vector<float>{
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
        -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_prelu_C_1_1) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/prelu_c_1_1.onnx"));

    Inputs inputs;
    // Shape{2, 3, 4, 5}
    inputs.emplace_back(std::vector<float>{
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.});

    // Shape{3, 1, 1}
    inputs.emplace_back(std::vector<float>{0, 1, 2});

    // Shape{2, 3, 4, 5}
    auto expected_output = std::vector<float>{
        -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
        -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_selu) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/selu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-5.99925954117548f, -5.99798722423258f, -5.99452870820667f, -5.98512748694000f, -5.95957231800549f},
              {-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30}},
             {{-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30},
              {33, 36, 39, 42, 45}},
             {{3, 3, 3, 3, 3},
              {-3.79272335297135f, -3.79272335297135f, -3.79272335297135f, -3.79272335297135f, -3.79272335297135f},
              {0, 0, 0, 0, 0},
              {6, 6, 6, 6, 6}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sigmoid) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sigmoid.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{0.00012339457598623f,
               0.00033535013046648f,
               0.00091105119440065f,
               0.00247262315663477f,
               0.00669285092428486f},
              {0.01798620996209160f, 0.04742587317756680f, 0.119202922022118f, 0.268941421369995f, 0.5f},
              {0.731058578630005f, 0.880797077977882f, 0.952574126822433f, 0.982013790037908f, 0.993307149075715f},
              {0.997527376843365f, 0.999088948805599f, 0.999664649869534f, 0.999876605424014f, 0.999954602131298f}},
             {{0.01798620996209160f, 0.04742587317756680f, 0.119202922022118f, 0.268941421369995f, 0.5f},
              {0.731058578630005f, 0.880797077977882f, 0.952574126822433f, 0.982013790037908f, 0.993307149075715f},
              {0.997527376843365f, 0.999088948805599f, 0.999664649869534f, 0.999876605424014f, 0.999954602131298f},
              {0.999983298578152f, 0.999993855825398f, 0.999997739675702f, 0.999999168471972f, 0.999999694097773f}},
             {{0.731058578630005f, 0.731058578630005f, 0.731058578630005f, 0.731058578630005f, 0.731058578630005f},
              {0.268941421369995f, 0.268941421369995f, 0.268941421369995f, 0.268941421369995f, 0.268941421369995f},
              {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
              {0.880797077977882f, 0.880797077977882f, 0.880797077977882f, 0.880797077977882f, 0.880797077977882f}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_tanh) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/tanh.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{-0.999999969540041f, -0.999999774929676f, -0.999998336943945f, -0.999987711650796f, -0.999909204262595f},
              {-0.999329299739067f, -0.995054753686731f, -0.964027580075817f, -0.761594155955765f, 0},
              {0.761594155955765f, 0.964027580075817f, 0.995054753686731f, 0.999329299739067f, 0.999909204262595f},
              {0.999987711650796f, 0.999998336943945f, 0.999999774929676f, 0.999999969540041f, 0.999999995877693f}},
             {{-0.999329299739067f, -0.995054753686731f, -0.964027580075817f, -0.761594155955765f, 0},
              {0.761594155955765f, 0.964027580075817f, 0.995054753686731f, 0.999329299739067f, 0.999909204262595f},
              {0.999987711650796f, 0.999998336943945f, 0.999999774929676f, 0.999999969540041f, 0.999999995877693f},
              {0.999999999442106f, 0.999999999924497f, 0.999999999989782f, 0.999999999998617f, 0.999999999999813f}},
             {{0.761594155955765f, 0.761594155955765f, 0.761594155955765f, 0.761594155955765f, 0.761594155955765f},
              {-0.761594155955765f, -0.761594155955765f, -0.761594155955765f, -0.761594155955765f, -0.761594155955765f},
              {0, 0, 0, 0, 0},
              {0.964027580075817f, 0.964027580075817f, 0.964027580075817f, 0.964027580075817f, 0.964027580075817f}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_thresholded_relu) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/thresholded_relu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>({{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>({{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}},
                                 {{0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
                                 {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_matmul_vec_ten3d) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/matmul_vec_ten3d.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f});
    inputs.emplace_back(test::NDArray<float, 3>{{{0.f}, {1.f}}, {{2.f}, {3.f}}, {{4.f}, {5.f}}}.get_vector());

    auto expected_output = test::NDArray<float, 2>{{1.f}, {3.f}, {5.f}}.get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softplus) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softplus.onnx"));

    // -1.0f, 0, 1.0f, 10.f,                    normal input values for activation
    // 100.0f, -100.0f, 1000.0f, -1000.0f,      input values that leads to exp() overflow
    // FLT_MIN, FLT_MIN / 16, -FLT_MIN / 16,    min, denorm, -denorm
    // FLT_MAX, -FLT_MAX,                       max, -max;
    Inputs inputs{std::vector<float>{-1.0f,
                                     0,
                                     1.0f,
                                     10.f,
                                     100.0f,
                                     -100.0f,
                                     1000.0f,
                                     -1000.0f,
                                     FLT_MIN,
                                     FLT_MIN / 16,
                                     -FLT_MIN / 16,
                                     FLT_MAX,
                                     -FLT_MAX}};

    const auto inf = std::numeric_limits<float>::infinity();
    std::vector<float> output{0.3132616579532623291f,
                              0.6931471824645996094f,
                              1.313261628150939941f,
                              10.0000457763671875f,
                              100.0f,
                              0.0f,
                              1000.0f,
                              0.0f,
                              0.6931471824645996094f,
                              0.6931471824645996094f,
                              0.6931471824645996094f,
                              inf,
                              0.0f};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softplus_infinity) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softplus.onnx"));

    std::vector<float> input(13, std::numeric_limits<float>::infinity());
    std::vector<float> expected_output(13, std::numeric_limits<float>::infinity());

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sum_opset8) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sum_opset8.onnx"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{1.0f, 2.0f, 3.0f});
    inputs.emplace_back(test::NDArray<float, 2>{{10.0f}, {20.0f}, {30.0f}}.get_vector());
    inputs.emplace_back(test::NDArray<float, 3>{{{100.0f}}, {{200.0f}}, {{300.0f}}}.get_vector());

    auto expected_output =
        test::NDArray<float, 3>{{{111.0f, 112.0f, 113.0f}, {121.0f, 122.0f, 123.0f}, {131.0f, 132.0f, 133.0f}},

                                {{211.0f, 212.0f, 213.0f}, {221.0f, 222.0f, 223.0f}, {231.0f, 232.0f, 233.0f}},

                                {{311.0f, 312.0f, 313.0f}, {321.0f, 322.0f, 323.0f}, {331.0f, 332.0f, 333.0f}}}
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmax_int32) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/argmax_int32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int64_t>({1, 1, 1, 1, 1, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_int32) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/argmin_int32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<std::int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<std::int64_t>({0, 0, 0, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmax_float) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/argmax_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({4.f, 0.1f, 2.f, 3.f, -3.f, 1.f, -0.9f, 0.f, 1.f, 2.f, 3.f, 0.f});
    test_case.add_expected_output<std::int64_t>({0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_float) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/argmin_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({4.f, 0.1f, 2.f, 3.f, -3.f, 1.f, -0.9f, 0.f, 1.f, 2.f, 3.f, 0.f});
    test_case.add_expected_output<std::int64_t>({1, 1, 0, 2});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmax_select_last_index) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/argmax_select_last_index.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3}, {1.f, 1.f, 1.f, 0.5f, 3.f, 4.f, 0.5f, 1.f, 1.1f, 0.f, 3.f, 0.f});
    test_case.add_expected_output<std::int64_t>(Shape{1, 3}, {0, 3, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_argmin_select_last_index) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/argmin_select_last_index.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3}, {1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.1f, 3.f, 3.f, 8.f});
    test_case.add_expected_output<std::int64_t>(Shape{4}, {2, 0, 1, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/top_k.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_10) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/top_k_opset_10.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_10_const_k) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/top_k_opset_10_const_k.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<float>(Shape{3, 3}, {3, 2, 1, 7, 6, 5, 11, 10, 9});       // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {3, 2, 1, 3, 2, 1, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/top_k_opset_11_const_k_smallest.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10});        // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {0, 1, 2, 0, 1, 2, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_top_k_opset_11_const_k_smallest_negative_axis) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/top_k_opset_11_const_k_smallest_negative_axis.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8});

    test_case.add_expected_output<float>(Shape{3, 3}, {0, 1, 2, 4, 5, 6, 8, 9, 10});        // values
    test_case.add_expected_output<std::int64_t>(Shape{3, 3}, {0, 1, 2, 0, 1, 2, 3, 2, 1});  // indices
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating_1D) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/top_k_repeating_1D.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>({1, 1, 2, 0, 2, 100});
    test_case.add_input<int64_t>({5});

    test_case.add_expected_output<int32_t>(Shape{5}, {100, 2, 2, 1, 1});
    test_case.add_expected_output<int64_t>(Shape{5}, {5, 2, 4, 0, 1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/top_k_repeating.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{3, 6}, {100, 1, 1, 2, 0, 2, 1, 2, 3, 4, 5, 6, 100, 1, 1, 2, 0, 2});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<int32_t>(Shape{3, 3}, {100, 2, 2, 6, 5, 4, 7, 2, 2});
    test_case.add_expected_output<int64_t>(Shape{3, 3}, {0, 3, 5, 5, 4, 3, 0, 2, 4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating_axis_0) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/top_k_repeating_axis_0.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{3, 6}, {100, 1, 1, 2, 0, 2, 1, 2, 3, 4, 5, 6, 7, 1, 2, 0, 2, 1});
    test_case.add_input<int64_t>({2});

    test_case.add_expected_output<int32_t>(Shape{2, 6}, {100, 2, 3, 4, 5, 6, 7, 1, 2, 2, 2, 2});
    test_case.add_expected_output<int64_t>(Shape{2, 6}, {0, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_top_k_repeating_unsorted) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/top_k_repeating_unsorted.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{3, 6}, {100, 1, 1, 2, 0, 2, 1, 2, 3, 4, 5, 6, 7, 1, 2, 0, 2, 1});
    test_case.add_input<int64_t>({3});

    test_case.add_expected_output<int32_t>(Shape{3, 3}, {1, 1, 0, 3, 2, 1, 1, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{3, 3}, {2, 1, 4, 2, 1, 0, 5, 1, 3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_acosh) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/acosh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3}, {1.0f, 2.5f, 4.3f});
    test_case.add_expected_output<float>(Shape{1, 3}, {0.0f, 1.5667993f, 2.13795861f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_asinh) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/asinh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-0.88137358f, 0.0f, 0.88137358f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_atanh) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/atanh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3}, {-0.9f, 0.0f, 0.9f});
    test_case.add_expected_output<float>(Shape{1, 3}, {-1.4722194f, 0.0f, 1.4722194f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sinh) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sinh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({-1.1752012f, 0.f, 1.1752012f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_cosh) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/cosh.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>({1.54308069f, 1.f, 1.54308069f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sign) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sign.onnx"));

    Inputs inputs{std::vector<float>{-std::numeric_limits<float>::infinity(),
                                     -3.141592f,
                                     0.0f,
                                     2.71828f,
                                     std::numeric_limits<float>::infinity()}};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output<float>({-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_one_hot_with_axis) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/one_hot_axis.onnx"));

    Inputs inputs{{1.0f, 9.0f, 2.0f, 4.0f}, {1.0f, 3.0f}};
    std::vector<float> expected_output{{1.0f, 1.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 3.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 3.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_one_hot_without_axis) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/one_hot_no_axis.onnx"));

    std::vector<std::vector<std::int64_t>> inputs{{0, 7, 8}, {2, 5}};
    std::vector<std::int64_t> expected_output{5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                              2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_where) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/where.onnx"));

    // conditions tensor - 3x3x3
    auto condition =
        std::vector<int>{{0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}};

    // 1x3 tensor of "1"
    auto x1 = std::vector<int>{1, 1, 1};
    // 3x1 tensor of "2"
    auto x2 = std::vector<int>{2, 2, 2};

    std::vector<std::vector<int>> inputs;
    inputs.push_back(std::move(condition));
    inputs.push_back(std::move(x1));
    inputs.push_back(std::move(x2));

    // y = 3x3x3
    std::vector<int> expected_output{2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_erf) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/erf.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 2>{
        {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {-3.141592f, 0.0f},
        {0.5f, 1.0f}}.get_vector());

    const std::vector<float> expected_output =
        test::NDArray<float, 2>{{-1.0f, 1.0f}, {-0.99999112f, 0.0f}, {0.52049988f, 0.84270079f}}.get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_erf_int32) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/erf_int32.onnx"));

    const std::vector<std::vector<int32_t>> inputs{
        {-std::numeric_limits<int32_t>::max(), -1, 0, 1, std::numeric_limits<int32_t>::max()}};

    const std::vector<int32_t> expected_output{-1, -1, 0, 1, 1};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shrink_float) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/shrink_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({-2.0f, -1.6f, -1.5f, -1.4f, -1.0f, 0.0f, 1.0f, 1.4f, 1.5f, 1.6f, 2.0f});
    test_case.add_expected_output<float>(Shape{11},
                                         {-1.5f, -1.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.1f, 1.5f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_shrink_int) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/shrink_int.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int>({-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5});
    test_case.add_expected_output<int>(Shape{11}, {-4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p1) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/lp_norm_p1.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.07142857f, 0.125f,  0.16666667f, 0.2f,        0.22727273f, 0.25f,   0.26923078f, 0.2857143f,
         0.3f,        0.3125f, 0.32352942f, 0.33333334f, 0.9285714f,  0.875f,  0.8333333f,  0.8f,
         0.77272725f, 0.75f,   0.7307692f,  0.71428573f, 0.7f,        0.6875f, 0.6764706f,  0.6666667f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_p2) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/lp_norm_p2.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f, 0.34570536f, 0.37139067f,
         0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,  0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f,
         0.9593655f,  0.9486833f,  0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/lp_norm_default.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.18257418f, 0.36514837f, 0.5477225f, 0.73029673f, 0.37904903f, 0.45485884f, 0.5306686f,  0.60647845f,
         0.42616236f, 0.47351375f, 0.5208651f, 0.5682165f,  0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,
         0.45862272f, 0.48560053f, 0.5125783f, 0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_lp_norm_default_dynamic) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/lp_norm_default_dynamic.onnx"));

    Shape data_shape{2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(data_shape, data);
    test_case.add_expected_output<float>(
        data_shape,
        {0.18257418f, 0.36514837f, 0.5477225f, 0.73029673f, 0.37904903f, 0.45485884f, 0.5306686f,  0.60647845f,
         0.42616236f, 0.47351375f, 0.5208651f, 0.5682165f,  0.4469492f,  0.48132992f, 0.51571065f, 0.5500913f,
         0.45862272f, 0.48560053f, 0.5125783f, 0.53955615f, 0.46609157f, 0.4882864f,  0.51048124f, 0.5326761f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_instance_normalization) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/instance_norm.onnx"));

    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    std::iota(std::begin(data), std::end(data), 1.f);

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(data);
    test_case.add_input<float>(std::vector<float>{2.134f, 3.256f});
    test_case.add_input<float>(std::vector<float>{0.765f, 1.055f});
    test_case.add_expected_output<float>(
        data_shape,
        {-2.6335807f,  -2.015657f, -1.3977331f, -0.77980936f, -0.16188562f, 0.45603812f, 1.0739619f,  1.6918856f,
         2.3098092f,   2.927733f,  3.5456567f,  4.1635804f,   -4.130463f,   -3.1876516f, -2.2448401f, -1.3020288f,
         -0.35921717f, 0.5835942f, 1.5264057f,  2.469217f,    3.4120288f,   4.35484f,    5.2976513f,  6.240463f});
    const size_t tolerance_bits = 3;
    test_case.run(tolerance_bits);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_instance_normalization_dynamic) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/instance_norm_dynamic.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{1.f, 2.f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 1, 1}, input_data);
    test_case.add_expected_output<float>(Shape{1, 3, 1, 1},
                                         {0.3341970741748809814f, 0.3321160078048706055f, 0.3407136797904968262f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_eye_like) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/eye_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{3, 4}, {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f});
    test_case.add_expected_output<float>(Shape{3, 4}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_0_batch_1) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reverse_sequence_time_0_batch_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.f, 4.f, 8.f, 12.f, 1.f, 5.f, 9.f, 13.f, 2.f, 6.f, 10.f, 14.f, 3.f, 7.f, 11.f, 15.f});
    test_case.add_input<int>({4, 3, 2, 1});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {3.f, 6.f, 9.f, 12.f, 2.f, 5.f, 8.f, 13.f, 1.f, 4.f, 10.f, 14.f, 0.f, 7.f, 11.f, 15.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_1_batch_0) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reverse_sequence_time_1_batch_0.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
    test_case.add_input<int>({1, 2, 3, 4});
    test_case.add_expected_output<float>(
        Shape{4, 4},
        {0.f, 1.f, 2.f, 3.f, 5.f, 4.f, 6.f, 7.f, 10.f, 9.f, 8.f, 11.f, 15.f, 14.f, 13.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_batch_axis) {
    EXPECT_THROW(
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reverse_sequence_incorrect_batch_axis.onnx")),
        ngraph_error)
        << "ReverseSequence batch_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_incorrect_time_axis) {
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                     SERIALIZED_ZOO,
                                                                     "onnx/reverse_sequence_incorrect_time_axis.onnx")),
                 ngraph_error)
        << "ReverseSequence time_axis attribute can only equal 0 or 1. Value of '2' is not "
           "accepted.";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reverse_sequence_time_and_batch_axis_equal) {
    EXPECT_THROW(
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/reverse_sequence_time_and_batch_axis_equal.onnx")),
        ngraph_error)
        << "ReverseSequence 'time_axis' and 'batch_axis' can't be equal.";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_matmul_float_type) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/matmul_float.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(std::vector<float>{0, 1, 2, 3, 4, 5});
    test_case.add_input<float>(std::vector<float>{0, 1});
    test_case.add_expected_output<float>(Shape{3, 1}, std::vector<float>{1, 3, 5});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mod_sign.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-4, 7, 5, 4, -7, 8});
    test_case.add_input<int32_t>({2, -3, 8, -2, 3, 5});
    test_case.add_expected_output<int32_t>(Shape{6}, {0, -2, 5, 0, 2, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_i64) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mod_sign_i64.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int64_t>({-4, 7, 5, 4, -7, 8});
    test_case.add_input<int64_t>({2, -3, 8, -2, 3, 5});
    test_case.add_expected_output<int64_t>(Shape{6}, {0, -2, 5, 0, 2, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_broadcast) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/mod_sign_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({3});
    test_case.add_expected_output<int32_t>(Shape{6}, {1, 0, 1, 0, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_f32) {
    try {
        const auto function = onnx_import::import_onnx_model(
            file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mod_sign_f32.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string("If the input type is floating point, then `fmod` attribute must be set to 1."));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mod_sign_fmod.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({22, -13, 8, -3, 7, 2});
    test_case.add_expected_output<int32_t>(Shape{6}, {-8, 3, 4, 0, -3, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod_broadcast) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/mod_sign_fmod_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-8, 3, 4, 9, -17, 1});
    test_case.add_input<int32_t>({3});
    test_case.add_expected_output<int32_t>(Shape{6}, {-2, 0, 1, 0, -2, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_sign_fmod_f32) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mod_sign_fmod_f32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
    test_case.add_input<float>({2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
    test_case.add_expected_output<float>(Shape{6}, {-0.10000038f, 0.39999962f, 5.f, 0.10000038f, -0.39999962f, 3.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mod_incorrect_fmod) {
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/mod_incorrect_fmod.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported value of 'fmod' attribute (should be: 0 or 1)"));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_param_i64_indices) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_nd_param_i64_indices.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<int64_t>({4, 3, 1, 7});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_const_i32_indices) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_nd_const_i32_indices.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_opset16_reduction_none) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_nd_opset16_reduction_none.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_input<int64_t>({4, 3, 1, 7});
    test_case.add_input<float>({9.f, 10.f, 11.f, 12.f});
    test_case.add_expected_output<float>(Shape{8}, {1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatterND_opset16_reduction_add) {
    EXPECT_THROW(onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                     SERIALIZED_ZOO,
                                                                     "onnx/scatter_nd_opset16_reduction_add.onnx")),
                 ngraph_error)
        << "Unsupported type of attribute: `reduction`. Only `none` is supported";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_float_1D) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/gather_float_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<float>(Shape{3},
        {   5, 6, 7 });
    test_case.add_input<int64_t>(Shape{2, 2},
        {   0, 1,
            1, 2    });
    test_case.add_expected_output<float>(Shape{2, 2},
        {   5, 6,
            6, 7    });
    // clang-format on

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_int32_3D_axis_1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/gather_int32_3D_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<int32_t>(Shape{2, 2, 2},
        {   1, 2,
            3, 4,

            5, 6,
            7, 8    });
    test_case.add_input<int32_t>(Shape{4, 1},
        {   0,
            1,
            1,
            0       });
    test_case.add_expected_output<int32_t>(Shape{2, 4, 1, 2},
        {   1, 2,
            3, 4,
            3, 4,
            1, 2,

            5, 6,
            7, 8,
            7, 8,
            5, 6     });
    // clang-format on

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_int8_3D_axis_neg_1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/gather_int8_3D_axis_neg_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<int8_t>(Shape{2, 2, 2},
        {   1, 2,
            3, 4,

            5, 6,
            7, 8            });
    test_case.add_input<int32_t>(Shape{4, 1},
        {   0, 1, 1, 0      });
    test_case.add_expected_output<int8_t>(Shape{2, 2, 4, 1},
        {   1, 2, 2, 1,
            3, 4, 4, 3,

            5, 6, 6, 5,
            7, 8, 8, 7      });
    // clang-format on

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_float_2D_neg_indices) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/gather_float_2D_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    // clang-format off
    test_case.add_input<float>(Shape{3, 3},
        {   0.0f, 0.1f, 0.2f,
            1.0f, 1.1f, 1.2f,
            2.0f, 2.1f, 2.2f   });
    test_case.add_input<int64_t>(Shape{2, 2},
        {   -1, -2,
            -3, -2      });
    test_case.add_expected_output<float>(Shape{3, 2, 2},
        {
            0.2f, 0.1f,
            0.0f, 0.1f,

            1.2f, 1.1f,
            1.0f, 1.1f,

            2.2f, 2.1f,
            2.0f, 2.1f    });
    // clang-format on

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_1D) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/gather_elements_float_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape{3}, {1, 2, 3});
    test_case.add_input<int64_t>(Shape{1}, {1});
    test_case.add_expected_output<float>(Shape{1}, {2});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int8_axis_1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/gather_elements_int8_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int8_t>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int32_t>(Shape{2, 2}, {0, 0, 1, 0});
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {1, 1, 4, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_int32_axis_0) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/gather_elements_int32_axis_0.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    test_case.add_input<int64_t>(Shape{2, 3}, {1, 2, 0, 2, 0, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 3}, {4, 8, 3, 7, 2, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_negative_axis) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/gather_elements_float_negative_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape{2, 2}, {1, 2, 3, 4});
    test_case.add_input<int64_t>(Shape{2, 2}, {1, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2, 2, 4, 3});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gather_elements_float_3D_axis_2) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/gather_elements_float_3D_axis_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape{2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    test_case.add_input<int64_t>(Shape{2, 2, 1}, {0, 1, 0, 1});
    test_case.add_expected_output<float>(Shape{2, 2, 1}, {1, 4, 5, 8});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gatherND_int32) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/gatherND_int32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({0, 1, 2, 3});
    test_case.add_input<int64_t>({1, 0});
    test_case.add_expected_output<int32_t>(Shape{2, 2}, {2, 3, 0, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_gatherND_float) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/gatherND_float.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
    test_case.add_input<int64_t>({0, 1, 1, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 4.f, 5.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_constant) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/pad_constant.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(Shape{3, 4},
                                         {0.f, 0.f, 1.f, 1.2f, 0.f, 0.f, 2.3f, 3.4f, 0.f, 0.f, 4.5f, 5.7f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_non_scalar_values) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/pad_non_scalar_values.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(Shape{3, 4},
                                         {44.f, 44.f, 1.f, 1.2f, 44.f, 44.f, 2.3f, 3.4f, 44.f, 44.f, 4.5f, 5.7f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_optional_constant) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/pad_optional_constant.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
    test_case.add_expected_output<float>(Shape{3, 4},
                                         {0.f, 0.f, 1.f, 1.2f, 0.f, 0.f, 2.3f, 3.4f, 0.f, 0.f, 4.5f, 5.7f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pad_constant_negative_begin_end) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/pad_negative_begin_end.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_input<int64_t>({-1, -1, -1, -1});
    test_case.add_expected_output<int32_t>(Shape{1, 2}, {6, 7});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pow_float32_float32) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/pow_float32_float32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});  // base
    test_case.add_input<float>({3.5f});                // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 11.313708f, 46.765373f, 128.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pow_float32_int32) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/pow_float32_int32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});  // base
    test_case.add_input<int>({3});                     // exponent

    test_case.add_expected_output<float>(Shape{1, 4}, {1.f, 8.f, 27.f, 64.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_pow_int32_float32) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/pow_int32_float32.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int>({1, 2, 3, 4});  // base
    test_case.add_input<float>({3.5f});      // exponent

    test_case.add_expected_output<int>(Shape{1, 4}, {1, 11, 46, 128});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_reciprocal) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/reciprocal.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    test_case.add_expected_output<float>(Shape{3, 2}, {1.f, 1 / 2.f, 1 / 3.f, 1 / 4.f, 1 / 5.f, 1 / 6.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_round) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/round.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.1f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.7f, -1.1f, -1.9f, -2.2f, -2.8f});
    test_case.add_expected_output<float>({0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -3.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_round_half_nearest_even) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/round_half_nearest_even.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.5f, 2.5f, -1.5f, -2.5f});
    test_case.add_expected_output<float>({0.f, 2.f, -2.f, -2.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter10) {
    const auto scatter_fn = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/scatter_opset10.onnx"));

    const Shape data_shape{2, 2};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<ov::op::v12::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({12.01f, 3.f, 4.f, 13.99f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_opset11) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_opset11.onnx"));

    const Shape data_shape{1, 5};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<ov::op::v12::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({1.f, 1.1f, 3.f, 2.1f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_opset16_reduction_none) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_opset16_reduction_none.onnx"));

    const Shape data_shape{1, 5};

    EXPECT_EQ(scatter_fn->get_output_size(), 1);
    EXPECT_EQ(scatter_fn->get_output_shape(0), data_shape);
    EXPECT_EQ(count_ops_of_type<ov::op::v12::ScatterElementsUpdate>(scatter_fn), 1);
    EXPECT_EQ(count_ops_of_type<op::v0::Constant>(scatter_fn), 4);

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({1.f, 1.1f, 3.f, 2.1f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_opset16_reduction_add) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_opset16_reduction_add.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_expected_output<float>({1.f, 3.1f, 3.f, 6.1f, 5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_default_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_default_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 11.f, 3.f, 12.f, 5.f, 6.f, 7.f, 13.f, 9.f, 14.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_none_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_none_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 11.f, 3.f, 12.f, 5.f, 6.f, 7.f, 13.f, 9.f, 14.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_none_neg_ind_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_none_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});               // Shape: (2, 5)
    test_case.add_input<int64_t>({-4, -2, -3, -1});                                                // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                                          // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 11.f, 3.f, 12.f, 5.f, 6.f, 7.f, 13.f, 9.f, 14.f});  // Shape: (2, 5)
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_add_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_add_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 13.f, 3.f, 16.f, 5.f, 6.f, 7.f, 21.f, 9.f, 24.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_add_neg_ind_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_add_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});               // Shape: (2, 5)
    test_case.add_input<int64_t>({-4, -2, -3, -1});                                                // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                                          // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 13.f, 3.f, 16.f, 5.f, 6.f, 7.f, 21.f, 9.f, 24.f});  // Shape: (2, 5)
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_mul_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_mul_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({11.f, 12.f, 13.f, 14.f});                             // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 22.f, 3.f, 48.f, 5.f, 6.f, 7.f, 104.f, 9.f, 140.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_min_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_min_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({0.f, 100.f, -1.f, 200.f});                            // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 0.f, 3.f, 4.f, 5.f, 6.f, 7.f, -1.f, 9.f, 10.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_scatter_elements_max_opset18) {
    const auto scatter_fn =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scatter_elements_max_opset18.onnx"));

    auto test_case = test::TestCase(scatter_fn, s_device);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});  // Shape: (2, 5)
    test_case.add_input<int64_t>({1, 3, 2, 4});                                       // Shape: (2, 2)
    test_case.add_input<float>({0.f, 100.f, -1.f, 200.f});                            // Shape: (2, 2)
    test_case.add_expected_output<float>({1.f, 2.f, 3.f, 100.f, 5.f, 6.f, 7.f, 8.f, 9.f, 200.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample6_nearest_infer) {
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/upsample6_nearest.onnx"));
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: nearest
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(input_shape,
        {   1.f, 2.f,
            3.f, 4.f    });
    test_case.add_expected_output<float>(expected_output_shape,
        {   1.f, 1.f, 1.f, 2.f, 2.f, 2.f,
            1.f, 1.f, 1.f, 2.f, 2.f, 2.f,
            3.f, 3.f, 3.f, 4.f, 4.f, 4.f,
            3.f, 3.f, 3.f, 4.f, 4.f, 4.f    });
    test_case.run();
    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample6_bilinear_infer) {
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/upsample6_bilinear.onnx"));
    // height_scale: 2.0
    // width_scale: 3.0
    // mode: bilinear
    const Shape input_shape          {1, 1, 2, 2};
    const Shape expected_output_shape{1, 1, 4, 6};

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(input_shape,
        {   1.f, 2.f,
            3.f, 4.f    });
    test_case.add_expected_output<float>(expected_output_shape,
        {   1.f,  4.f/3,  5.f/3, 2.f, 2.f, 2.f,
            2.f,  7.f/3,  8.f/3, 3.f, 3.f, 3.f,
            3.f, 10.f/3, 11.f/3, 4.f, 4.f, 4.f,
            3.f, 10.f/3, 11.f/3, 4.f, 4.f, 4.f  });
    test_case.run();
    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample6_dynamic) {
    // clang-format off
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/upsample6_dynamic.onnx"));
    // height_scale: 1.5
    // width_scale: 2.5
    // mode: nearest
    //
    //  X ───╤══> Reshape ──R──> Upsample ──> Y
    //  S ───┘

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape {4},                      // X
        {   1.f, 2.f, 3.f, 4.f  });
    test_case.add_input<int64_t>(Shape {4},    {1, 1, 2, 2});  // S
    test_case.add_expected_output<float>(Shape {1, 1, 3, 5},   // Y
        {   1.f, 1.f, 1.f, 2.f, 2.f,
            1.f, 1.f, 1.f, 2.f, 2.f,
            3.f, 3.f, 3.f, 4.f, 4.f    });
    test_case.run();
    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample8_nearest_infer) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/upsample8_nearest.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample8_linear_infer) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/upsample8_linear.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Scales attribute values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.5f, 4.0f, 4.0f, 3.0f, 3.5f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_nearest_infer) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/upsample9_scales_const_nearest.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 3.0}
    // mode: nearest

    const Shape expected_output_shape{1, 1, 4, 6};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(expected_output_shape,
                                         {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_upsample9_scales_const_linear_infer) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/upsample9_scales_const_linear.onnx"));

    // Input data shape (1, 1, 2, 2)
    // Input const scales values {1.0, 1.0, 2.0, 2.0}
    // mode: linear

    const Shape expected_output_shape{1, 1, 4, 4};
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(
        expected_output_shape,
        {1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.5f, 4.0f, 4.0f, 3.0f, 3.5f, 4.0f, 4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_image_scaler) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/image_scaler.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 2}, {12.0f, 14.0f, 16.0f, 18.0f, 21.0f, 41.0f, 61.0f, 81.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_single) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/size_op_single.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    test_case.add_expected_output<int64_t>(Shape{}, {6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_graph_end) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/size_op_graph_end.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<int64_t>(Shape{}, {4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_graph_middle) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/size_op_graph_middle.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f});
    test_case.add_expected_output<float>(Shape{}, {4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_size_op_on_input_graph_middle) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/size_op_on_input_graph_middle.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 2, 4, 1, 3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                                      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_expected_output<float>(Shape{1, 2, 4, 1, 3},
                                         {24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f,
                                          24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_empty_initializers_handling) {
    // int this test the "scales" input of the Resize operator is set to an empty initializer
    // this input should be ignored since the "sizes" optional input is provided
    // and the inference should use the data from the latter
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/empty_initializers_handling.onnx"));

    const Shape expected_output_shape{2, 1, 4, 8};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{2.0f, 4.0f, 1.0f, 3.0f, 7.0f, 8.0f, 9.0f, 6.0f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        expected_output_shape,
        {2.0f, 2.5f,  3.0f, 3.5f,  4.0f, 4.0f, 4.0f, 4.0f, 1.5f, 2.0f,  2.5f, 3.0f,  3.5f, 3.5f, 3.5f, 3.5f,
         1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.0f, 3.0f, 3.0f,
         7.0f, 7.25f, 7.5f, 7.75f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 7.75f, 7.5f, 7.25f, 7.0f, 7.0f, 7.0f, 7.0f,
         9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f, 9.0f, 8.25f, 7.5f, 6.75f, 6.0f, 6.0f, 6.0f, 6.0f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_roi_align_f32) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/roi_align_f32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.});

    test_case.add_input<float>(
        {7., 5., 7., 5., -15., -15., -15., -15., -10., 21., -10., 21., 13., 8., 13., 8., -14., 19., -14., 19.});

    test_case.add_input<int32_t>({0, 0, 0, 0, 0});
    test_case.add_expected_output<float>(
        Shape{5, 3, 3, 4},
        {2.95833f, 3.20833f, 3.45833f, 3.70833f, 4.625f,   4.875f,   5.125f,   5.375f,   6.29167f, 6.54167f, 6.79167f,
         7.04167f, 27.9583f, 28.2083f, 28.4583f, 28.7083f, 29.625f,  29.875f,  30.125f,  30.375f,  31.2917f, 31.5417f,
         31.7917f, 32.0417f, 52.9583f, 53.2083f, 53.4583f, 53.7083f, 54.625f,  54.875f,  55.125f,  55.375f,  56.2917f,
         56.5417f, 56.7917f, 57.0417f, 0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
         0.f,      0.f,      0.f,      0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
         25.f,     25.f,     25.f,     25.f,     25.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
         50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     7.39583f, 7.39583f, 7.42708f, 7.64583f, 9.0625f,
         9.0625f,  9.09375f, 9.3125f,  10.7292f, 10.7292f, 10.7604f, 10.9792f, 32.3958f, 32.3958f, 32.4271f, 32.6458f,
         34.0625f, 34.0625f, 34.0938f, 34.3125f, 35.7292f, 35.7292f, 35.7604f, 35.9792f, 57.3958f, 57.3958f, 57.4271f,
         57.6458f, 59.0625f, 59.0625f, 59.0938f, 59.3125f, 60.7292f, 60.7292f, 60.7604f, 60.9792f, 4.27083f, 4.52083f,
         4.77083f, 5.02083f, 5.9375f,  6.1875f,  6.4375f,  6.6875f,  7.60417f, 7.85417f, 8.10417f, 8.35417f, 29.2708f,
         29.5208f, 29.7708f, 30.0208f, 30.9375f, 31.1875f, 31.4375f, 31.6875f, 32.6042f, 32.8542f, 33.1042f, 33.3542f,
         54.2708f, 54.5208f, 54.7708f, 55.0208f, 55.9375f, 56.1875f, 56.4375f, 56.6875f, 57.6042f, 57.8542f, 58.1042f,
         58.3542f, 6.77083f, 6.77083f, 6.77083f, 6.80208f, 8.4375f,  8.4375f,  8.4375f,  8.46875f, 10.1042f, 10.1042f,
         10.1042f, 10.1354f, 31.7708f, 31.7708f, 31.7708f, 31.8021f, 33.4375f, 33.4375f, 33.4375f, 33.4688f, 35.1042f,
         35.1042f, 35.1042f, 35.1354f, 56.7708f, 56.7708f, 56.7708f, 56.8021f, 58.4375f, 58.4375f, 58.4375f, 58.4688f,
         60.1042f, 60.1042f, 60.1042f, 60.1354f});
    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_roialign16_avg_out_half_pixel) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/roialign16_avg_out_half_pixel.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.1f,   2.2f,   3.3f,   4.4f,   5.5f,   6.6f,   7.7f,   8.8f,   9.9f,   11.f,   12.1f,  13.2f,  14.3f,  15.4f,
         16.5f,  17.6f,  18.7f,  19.8f,  20.9f,  22.f,   23.1f,  24.2f,  25.3f,  26.4f,  27.5f,  28.6f,  29.7f,  30.8f,
         31.9f,  33.f,   34.1f,  35.2f,  36.3f,  37.4f,  38.5f,  39.6f,  40.7f,  41.8f,  42.9f,  44.f,   45.1f,  46.2f,
         47.3f,  48.4f,  49.5f,  50.6f,  51.7f,  52.8f,  53.9f,  55.f,   56.1f,  57.2f,  58.3f,  59.4f,  60.5f,  61.6f,
         62.7f,  63.8f,  64.9f,  66.f,   67.1f,  68.2f,  69.3f,  70.4f,  71.5f,  72.6f,  73.7f,  74.8f,  75.9f,  77.f,
         78.1f,  79.2f,  80.3f,  81.4f,  82.5f,  83.6f,  84.7f,  85.8f,  86.9f,  88.f,   89.1f,  90.2f,  91.3f,  92.4f,
         93.5f,  94.6f,  95.7f,  96.8f,  97.9f,  99.f,   100.1f, 101.2f, 102.3f, 103.4f, 104.5f, 105.6f, 106.7f, 107.8f,
         108.9f, 110.f,  111.1f, 112.2f, 113.3f, 114.4f, 115.5f, 116.6f, 117.7f, 118.8f, 119.9f, 121.f,  122.1f, 123.2f,
         124.3f, 125.4f, 126.5f, 127.6f, 128.7f, 129.8f, 130.9f, 132.f,  133.1f, 134.2f, 135.3f, 136.4f, 137.5f, 138.6f,
         139.7f, 140.8f, 141.9f, 143.f,  144.1f, 145.2f, 146.3f, 147.4f, 148.5f, 149.6f, 150.7f, 151.8f, 152.9f, 154.f,
         155.1f, 156.2f, 157.3f, 158.4f, 159.5f, 160.6f, 161.7f, 162.8f, 163.9f, 165.f,  166.1f, 167.2f, 168.3f, 169.4f,
         170.5f, 171.6f, 172.7f, 173.8f, 174.9f, 176.f,  177.1f, 178.2f, 179.3f, 180.4f, 181.5f, 182.6f, 183.7f, 184.8f,
         185.9f, 187.f,  188.1f, 189.2f, 190.3f, 191.4f, 192.5f, 193.6f, 194.7f, 195.8f, 196.9f, 198.f,  199.1f, 200.2f,
         201.3f, 202.4f, 203.5f, 204.6f, 205.7f, 206.8f, 207.9f, 209.f,  210.1f, 211.2f, 212.3f, 213.4f, 214.5f, 215.6f,
         216.7f, 217.8f, 218.9f, 220.f,  221.1f, 222.2f, 223.3f, 224.4f, 225.5f, 226.6f, 227.7f, 228.8f, 229.9f, 231.f,
         232.1f, 233.2f, 234.3f, 235.4f, 236.5f, 237.6f});

    test_case.add_input<float>({0.f, 0.f, 0.75f, 2.2f, 1.2f, 0.5f, 2.8f, 1.9f, 0.f, 3.f, 0.f, 3.f});

    test_case.add_input<int64_t>({0, 2, 1});
    test_case.add_expected_output<float>(
        Shape{3, 2, 4, 4},
        {2.145f,     2.42f,      2.6950002f, 2.9700003f, 3.96f,      4.235f,     4.51f,      4.7850003f, 5.775f,
         6.05f,      6.325f,     6.6000004f, 7.59f,      7.8650007f, 8.14f,      8.415001f,  41.745003f, 42.019997f,
         42.295f,    42.57f,     43.56f,     43.835f,    44.11f,     44.385002f, 45.375f,    45.65f,     45.925003f,
         46.200005f, 47.190002f, 47.465004f, 47.74f,     48.015f,    162.77249f, 163.0475f,  163.32251f, 163.5975f,
         164.42252f, 164.69751f, 164.9725f,  165.2475f,  166.07251f, 166.3475f,  166.6225f,  166.8975f,  167.72249f,
         167.9975f,  168.27249f, 168.5475f,  202.3725f,  202.6475f,  202.9225f,  203.19751f, 204.02252f, 204.2975f,
         204.57251f, 204.8475f,  205.6725f,  205.94751f, 206.2225f,  206.4975f,  207.32251f, 207.5975f,  207.8725f,
         208.1475f,  91.162506f, 91.4375f,   91.7125f,   91.9875f,   92.8125f,   93.0875f,   93.3625f,   93.6375f,
         94.4625f,   94.7375f,   95.0125f,   95.28749f,  96.1125f,   96.3875f,   96.6625f,   96.9375f,   130.76251f,
         131.0375f,  131.3125f,  131.5875f,  132.4125f,  132.6875f,  132.9625f,  133.2375f,  134.0625f,  134.33751f,
         134.6125f,  134.88751f, 135.7125f,  135.9875f,  136.26251f, 136.53749f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_roialign16_avg_half_pixel) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/roialign16_avg_half_pixel.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        {1.1f,     2.2f,   3.3f,   4.4f,   5.5f,   6.6f,   7.7f,   8.8f,   9.9f,   11.f,   12.1f,  13.2f,  14.3f,
         15.4f,    16.5f,  17.6f,  18.7f,  19.8f,  20.9f,  22.f,   23.1f,  24.2f,  25.3f,  26.4f,  27.5f,  28.6f,
         29.7f,    30.8f,  31.9f,  33.f,   34.1f,  35.2f,  36.3f,  37.4f,  38.5f,  39.6f,  40.7f,  41.8f,  42.9f,
         44.f,     45.1f,  46.2f,  47.3f,  48.4f,  49.5f,  50.6f,  51.7f,  52.8f,  53.9f,  55.f,   56.1f,  57.2f,
         58.3f,    59.4f,  60.5f,  61.6f,  62.7f,  63.8f,  64.9f,  66.f,   67.1f,  68.2f,  69.3f,  70.4f,  71.5f,
         72.6f,    73.7f,  74.8f,  75.9f,  77.f,   78.1f,  79.2f,  80.3f,  81.4f,  82.5f,  83.6f,  84.7f,  85.8f,
         86.9f,    88.f,   89.1f,  90.2f,  91.3f,  92.4f,  93.5f,  94.6f,  95.7f,  96.8f,  97.9f,  99.f,   100.1f,
         101.2f,   102.3f, 103.4f, 104.5f, 105.6f, 106.7f, 107.8f, 108.9f, 110.f,  111.1f, 112.2f, 113.3f, 114.4f,
         115.5f,   116.6f, 117.7f, 118.8f, 119.9f, 121.f,  122.1f, 123.2f, 124.3f, 125.4f, 126.5f, 127.6f, 128.7f,
         129.8f,   130.9f, 132.f,  133.1f, 134.2f, 135.3f, 136.4f, 137.5f, 138.6f, 139.7f, 140.8f, 141.9f, 143.f,
         144.1f,   145.2f, 146.3f, 147.4f, 148.5f, 149.6f, 150.7f, 151.8f, 152.9f, 154.f,  155.1f, 156.2f, 157.3f,
         158.4f,   159.5f, 160.6f, 161.7f, 162.8f, 163.9f, 165.f,  166.1f, 167.2f, 168.3f, 169.4f, 170.5f, 171.6f,
         172.7f,   173.8f, 174.9f, 176.f,  177.1f, 178.2f, 179.3f, 180.4f, 181.5f, 182.6f, 183.7f, 184.8f, 185.9f,
         187.198f, 188.1f, 189.2f, 190.3f, 191.4f, 192.5f, 193.6f, 194.7f, 195.8f, 196.9f, 198.f,  199.1f, 200.2f,
         201.3f,   202.4f, 203.5f, 204.6f, 205.7f, 206.8f, 207.9f, 209.f,  210.1f, 211.2f, 212.3f, 213.4f, 214.5f,
         215.6f,   216.7f, 217.8f, 218.9f, 220.f,  221.1f, 222.2f, 223.3f, 224.4f, 225.5f, 226.6f, 227.7f, 228.8f,
         229.9f,   231.f,  232.1f, 233.2f, 234.3f, 235.4f, 236.5f, 237.6f});

    test_case.add_input<float>({0.f, 0.f, 0.75f, 2.2f, 1.2f, 0.5f, 2.8f, 1.9f, 0.f, 3.f, 0.f, 3.f});

    test_case.add_input<int64_t>({0, 2, 1});
    test_case.add_expected_output<float>(
        Shape{3, 2, 4, 4},
        {1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       1.1f,       2.3375f,
         2.3375f,    2.3375f,    2.3375f,    4.1525f,    4.1525f,    4.1525f,    4.1525f,    40.7f,      40.7f,
         40.7f,      40.7f,      40.7f,      40.7f,      40.7f,      40.7f,      41.9375f,   41.9375f,   41.9375f,
         41.9375f,   43.7525f,   43.7525f,   43.7525f,   43.7525f,   159.72f,    159.94f,    160.16f,    160.38f,
         159.90562f, 160.12563f, 160.34563f, 160.56563f, 160.9575f,  161.1775f,  161.3975f,  161.61751f, 162.1125f,
         162.3325f,  162.55249f, 162.77249f, 199.32f,    199.54001f, 199.76001f, 199.97998f, 199.50562f, 199.72563f,
         199.94562f, 200.16562f, 200.5575f,  200.7775f,  200.9975f,  201.2175f,  201.7125f,  201.93251f, 202.1525f,
         202.37251f, 86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,
         86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      86.9f,      126.5f,
         126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f,
         126.5f,     126.5f,     126.5f,     126.5f,     126.5f,     126.5f});
    test_case.run_with_tolerance_as_fp(0.01f);
}

OPENVINO_TEST(${BACKEND_NAME}, quant_dequant_pattern) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/quant_dequant_pattern.onnx"));
    auto test_case = test::TestCase(function, s_device);
    // scale == 3.0
    // zero point == 10
    test_case.add_input<float>({9.0f, 10.0f, 15.0f, 20.0f, 30.0f});
    test_case.add_input<float>({1.f});
    test_case.add_expected_output<float>(Shape{5}, {9.0f, 9.0f, 15.0f, 21.0f, 30.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, quant_dequant_pattern_axis) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/quant_dequant_pattern_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);
    // axis = 1
    // scale == {2.0, 3.0, 4.0}
    // zero point == {10, 20, 30}
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 100.0f});
    test_case.add_expected_output<float>(Shape{3, 3}, {0.f, 3.f, 4.f, 10.f, 21.f, 32.f, 40.f, 51.f, 100.f});
    test_case.add_input<float>({1.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_0D) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softmax_0D.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({3.141592f});
    test_case.add_expected_output<float>({0.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax_1D) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/logsoftmax_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061f, -1.407606f, -0.407606f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_1D) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/logsoftmax13_1D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-1.0f, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.4076061f, -1.407606f, -0.407606f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/logsoftmax13_2D.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({0.0f, 1.0f, 2.0f, 3.0f, 10000.f, 10001.f, 10002.f, 10003.f});
    test_case.add_expected_output<float>(
        Shape{2, 4},
        {-3.4401896f, -2.4401896f, -1.4401896f, -0.44018966f, -3.4401896f, -2.4401896f, -1.4401896f, -0.44018966f});
    test_case.run_with_tolerance_as_fp();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_logsoftmax13_2D_reshape) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/logsoftmax13_2D.onnx"));
    InferenceEngine::CNNNetwork net(function);
    InferenceEngine::ICNNNetwork::InputShapes shapes = {};
    InferenceEngine::SizeVector shape = {1, 1, 4000};
    shapes[net.getInputsInfo().begin()->first] = shape;
    EXPECT_NO_THROW(net.reshape(shapes));
    ASSERT_EQ(shape, net.getOutputsInfo().begin()->second->getDims());
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_hard_sigmoid) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/hard_sigmoid.onnx"));

    const auto inf = std::numeric_limits<float>::infinity();
    const auto neg_inf = -std::numeric_limits<float>::infinity();

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({inf, neg_inf, 0.0f, 1.0f});
    test_case.add_expected_output<float>(Shape{4}, {1.0f, 0.0f, 0.5f, 0.699999988079071f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mul_v6.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axis_1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/mul_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {3.0f, 6.0f, 9.0f, 12.0f, 20.0f, 24.0f, 28.0f, 32.0f, 45.0f, 50.0f, 55.0f, 60.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_axes_1_2) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/mul_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), -1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape,
                                         {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v6_broadcast_no_axis) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/mul_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 6.0f, 9.0f, 12.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v7) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mul_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {3.0f, 8.0f, 15.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_mul_v7_broadcast) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mul_v7_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {3.0f, 8.0f, 15.0f, 12.0f, 20.0f, 30.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axis_1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/add_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f, 17.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_axes_1_2) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/add_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape, {3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v6_broadcast_no_axis) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/add_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_add_v7) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/add_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(Shape{3}, {4.0f, 6.0f, 8.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axis_1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/sub_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape,
                                         {-2.0f, -1.0f, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_axes_1_2) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/sub_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 0.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(shape,
                                         {-3.f, -3.f, -4.f, -4.f, -5.f, -5.f, -6.f, -6.f, -7.f, -7.f, -8.f, -8.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v6_broadcast_no_axis) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/sub_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -1.0f, 0.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v7) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sub_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {-2.0f, -6.0f, -4.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_sub_v7_broadcast) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/sub_v7_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {-2.0f, -2.0f, -2.0f, 1.0f, 1.0f, 1.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axis_1) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/div_v6_broadcast_axis_1.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(
        shape,
        {0.3333333f, 0.6666666f, 1.0f, 1.333333f, 1.25f, 1.5f, 1.75f, 2.0f, 1.8f, 2.0, 2.2f, 2.4f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_axes_1_2) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/div_v6_broadcast_axes_1_2.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 3, 2, 2};
    std::vector<float> A(shape_size(shape), 840.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(
        shape,
        {280.f, 280.f, 210.f, 210.f, 168.f, 168.f, 140.f, 140.f, 120.f, 120.f, 105.f, 105.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v6_broadcast_no_axis) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/div_v6_broadcast_no_axis.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{2, 2};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({2.0f});
    test_case.add_expected_output<float>(shape, {0.5f, 1.0f, 1.5f, 2.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v7) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/div_v7.onnx"));
    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 2.0f, 3.0f});
    test_case.add_input<float>({3.0f, 8.0f, 7.0f});
    test_case.add_expected_output<float>(Shape{3}, {0.3333333f, 0.25f, 0.4285714f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_div_v7_broadcast) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/div_v7_broadcast.onnx"));
    auto test_case = test::TestCase(function, s_device);

    Shape shape{1, 2, 3};
    std::vector<float> A(shape_size(shape));
    std::iota(A.begin(), A.end(), 1.f);
    test_case.add_input<float>(A);
    test_case.add_input<float>({3.0f, 4.0f, 5.0f});
    test_case.add_expected_output<float>(shape, {0.3333333f, 0.5f, 0.6f, 1.3333333f, 1.25f, 1.2f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_dangling_parameter) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/dangling_parameter.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({-1.0f, 2.0f, -3.0f});
    test_case.add_expected_output<float>(Shape{3}, {1.0f, 2.0f, 3.0f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_inbounds) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/test_clip_inbounds.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<int32_t> data{-1, 0, 1, -9999, 9999};
    test_case.add_input<int32_t>(data);
    test_case.add_expected_output<int32_t>(Shape{data.size()}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_no_min_no_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max_inf) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_no_min_no_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  static_cast<float>(std::numeric_limits<float>::max()),
                                  std::numeric_limits<float>::min(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::lowest(),
                                  0.f,
                                  -1.f};

    const std::vector<float> expected_output{std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::lowest(),
                                             std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::min(),
                                             std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::lowest(),
                                             0.f,
                                             -1.f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, expected_output);
    test_case.run_with_tolerance_as_fp(0.f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_set_max) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_no_min_set_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> max_val{2.01f};
    const std::vector<float> output{-1.6f, -0.1f, 2.01f, 0.f, -10.f, 1.99f, 2.01f, 2.01f};

    test_case.add_input<float>(data);
    test_case.add_input<float>(max_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_no_max) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_set_min_no_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> min_val{-1.59f};
    const std::vector<float> output{-1.59f, -0.1f, 10.f, 0.f, -1.59f, 1.99f, 2.015f, 3.f};

    test_case.add_input<float>(data);
    test_case.add_input<float>(min_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_no_max_int64) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_no_min_no_max_int64.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<int64_t> data{INT64_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};

    test_case.add_input<int64_t>(data);

    test_case.add_expected_output<int64_t>(Shape{2, 4}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_no_min_set_max_int64) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_no_min_set_max_int64.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<int64_t> data{INT64_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};
    const std::vector<int64_t> max_val{INT32_MAX};
    const std::vector<int64_t> output{INT32_MAX, INT64_MIN, INT32_MAX, INT32_MIN, 0, -1, 1, 0};

    test_case.add_input<int64_t>(data);
    test_case.add_input<int64_t>(max_val);

    test_case.add_expected_output<int64_t>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_no_max_initializers) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_set_min_no_max_initializers.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> output{-1.59f, -0.1f, 10.f, 0.f, -1.59f, 1.99f, 2.015f, 3.f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_set_max) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_set_min_set_max.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> min_val{-1.59f};
    const std::vector<float> max_val{2.01f};
    const std::vector<float> output{-1.59f, -0.1f, 2.01f, 0.f, -1.59f, 1.99f, 2.01f, 2.01f};

    test_case.add_input<float>(data);
    test_case.add_input<float>(min_val);
    test_case.add_input<float>(max_val);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_clip_set_min_set_max_initializers) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/clip_set_min_set_max_initializers.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data{-1.6f, -0.1f, 10.f, 0.f, -10.f, 1.99f, 2.015f, 3.f};
    const std::vector<float> output{-1.59f, -0.1f, 2.01f, 0.f, -1.59f, 1.99f, 2.01f, 2.01f};

    test_case.add_input<float>(data);

    test_case.add_expected_output<float>(Shape{2, 4}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_mvn_v6) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/mvn_v6.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0.8439683f,  0.5665144f, 0.05836735f, 0.02916367f, 0.12964272f, 0.5060197f, 0.79538304f,
                                0.9411346f,  0.9546573f, 0.17730942f, 0.46192095f, 0.26480448f, 0.6746842f, 0.01665257f,
                                0.62473077f, 0.9240844f, 0.9722341f,  0.11965699f, 0.41356155f, 0.9129373f, 0.59330076f,
                                0.81929934f, 0.7862604f, 0.11799799f, 0.69248444f, 0.54119414f, 0.07513223f});
    test_case.add_expected_output<float>(
        Shape{3, 3, 3, 1},
        {1.3546423f,  0.33053496f, -1.5450814f,  -1.2106764f,  -0.8925952f,  0.29888135f, 0.38083088f,
         0.81808794f, 0.85865635f, -1.1060555f,  -0.05552877f, -0.78310335f, 0.83281356f, -1.250282f,
         0.67467856f, 0.7669372f,  0.9113869f,   -1.6463585f,  -0.23402764f, 1.6092131f,  0.42940593f,
         1.2906139f,  1.1860244f,  -0.92945826f, 0.0721334f,   -0.38174f,    -1.7799333f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_no_return_mask) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/dropout1_no_training_no_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout1_no_training_return_mask) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/dropout1_no_training_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(Shape{3, 4, 5},
                                           std::vector<int32_t>(3 * 4 * 5, 1));  // // bool converted to i32
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout7_no_return_mask) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/dropout7_no_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_no_return_mask) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/dropout12_no_training_no_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_no_training_return_mask) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/dropout12_no_training_return_mask.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const std::vector<float> data(3 * 4 * 5, 2.0f);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{3, 4, 5}, data);
    test_case.add_expected_output<int32_t>(Shape{3, 4, 5},
                                           std::vector<int32_t>(3 * 4 * 5, 1));  // bool converted to i32
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_no_traning_no_const_rato) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/dropout12_no_traning_no_const_rato.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({1, 2, 3, 4});
    // test_case.add_input<float>(Shape{}, {0.5}); // ratio input is ignored

    test_case.add_expected_output<float>(Shape{1, 4}, {1., 2., 3., 4.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_training_mode) {
    try {
        auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                            SERIALIZED_ZOO,
                                                                            "onnx/dropout12_training_mode.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Training mode is not supported for Dropout op"));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_dropout12_not_const_training_mode) {
    try {
        auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/dropout12_not_const_training_mode.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Non-constant training_mode input is not supported."));
    } catch (...) {
        FAIL() << "Expected ngraph_error exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_multiple_slices_last_layer) {
    std::vector<float> data(1 * 30 * 320 * 320);
    std::fill(data.begin(), data.end(), 1.f);

    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/multiple_slices_last_layer.onnx"));
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> o1(1 * 320 * 320 * 21);
    std::fill(o1.begin(), o1.end(), 1.f);

    std::vector<float> o2(1 * 320 * 320 * 9);
    std::fill(o2.begin(), o2.end(), 1.f);

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 21}, o1);
    test_case.add_expected_output<float>(Shape{1, 320, 320, 9}, o2);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_slice_const_axes_source) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/slice_const_axes_source.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {2.f, 3.f, 6.f, 7.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_softmax_crossentropy_loss_mean) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/softmax_crossentropy_loss_mean.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({0.54881352186203f,
                                0.7151893377304077f,
                                0.6027633547782898f,
                                0.5448831915855408f,
                                0.42365479469299316f,
                                0.6458941102027893f,
                                0.4375872015953064f,
                                0.891772985458374f,
                                0.9636627435684204f,
                                0.3834415078163147f,
                                0.7917250394821167f,
                                0.5288949012756348f,
                                0.5680445432662964f,
                                0.9255966544151306f,
                                0.07103605568408966f});
    test_case.add_input<int64_t>({1, 4, 3});
    test_case.add_expected_output<float>(Shape{}, {1.561384797096252441f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_negativelog_likelihood_loss) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/negativelog_likelihood_loss.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({
        0.54881352186203f,    0.7151893377304077f,   0.6027633547782898f, 0.5448831915855408f, 0.42365479469299316f,
        0.6458941102027893f,  0.4375872015953064f,   0.891772985458374f,  0.9636627435684204f, 0.3834415078163147f,
        0.7917250394821167f,  0.5288949012756348f,   0.5680445432662964f, 0.9255966544151306f, 0.07103605568408966f,
        0.08712930232286453f, 0.020218396559357643f, 0.832619845867157f,  0.7781567573547363f, 0.8700121641159058f,
        0.978618323802948f,   0.7991585731506348f,   0.4614793658256531f, 0.7805292010307312f, 0.11827442795038223f,
        0.6399210095405579f,  0.14335328340530396f,  0.9446688890457153f, 0.5218483209609985f, 0.4146619439125061f,
    });
    test_case.add_input<int64_t>({3, 3, 2, 4, 2, 0});
    test_case.add_expected_output<float>(Shape{}, {-0.531306922435760498f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_input_as_shape_default_value) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/constant_fill_input_as_shape_default_value.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{1, 2, 3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_input_as_shape_u8_type) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/constant_fill_input_as_shape_u8_type.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint8_t>(Shape{3, 1, 2}, {3, 3, 3, 3, 3, 3});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_extra_shape) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_fill_extra_shape.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3, 1, 2, 2, 1}, std::vector<float>(12, 3.0f));
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_fill_shape_attribute) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_fill_shape_attribute.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int32_t>(Shape{2, 3, 4}, std::vector<int32_t>(24, 5));
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_float_tensor) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_float_tensor.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 3}, {0.0f, 0.5f, 1.f, 1.5f, 2.f, 2.5f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_bfloat_tensor) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_bfloat_tensor.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<bfloat16>(Shape{2, 3}, {0.f, 5.f, 10.f, 15.f, 20.f, 25.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_float_scalar) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_float_scalar.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{}, {0.5f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_float_array) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_float_array.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3}, {0.5f, 1.f, 1.5f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_integer_scalar) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_integer_scalar.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<std::int64_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_integer_array) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_integer_array.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<std::int64_t>(Shape{3}, {0, 1, 2});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x2) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 2}, {0.f, 5.f, 0.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_float_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_3x4_linearized_indices) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/constant_sparse_tensor_float_3x4_linearized_indices.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int32_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_int32_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int32_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int64_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_int64_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int64_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_boolean_3x4) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/constant_sparse_tensor_boolean_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<bool>(Shape{3, 4}, {1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float16_3x4) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/constant_sparse_tensor_float16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<ngraph::float16>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_double_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_double_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<double>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int8_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_int8_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int8_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_int16_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_int16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int16_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint8_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_uint8_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint8_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint16_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_uint16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint16_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint32_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_uint32_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint32_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_uint64_3x4) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_uint64_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<uint64_t>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_bfloat16_3x4) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/constant_sparse_tensor_bfloat16_3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<ngraph::bfloat16>(Shape{3, 4}, {1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 0});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_8x17) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_sparse_tensor_float_8x17.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(
        Shape{8, 17},
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x3x4) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/constant_sparse_tensor_float_2x3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 3, 4}, {1.f, 0.f, 0.f, 8.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f,
                                                          0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 3.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_sparse_tensor_float_2x2x3x4) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/constant_sparse_tensor_float_2x2x3x4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(
        Shape{2, 2, 3, 4},
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 3.f, 0.f, 0.f, 0.f, 0.f,
         0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 8.f, 0.f, 1.f, 2.f, 0.f,
         0.f, 0.f, 3.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_einsum_sum) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/einsum_sum.onnx"));
    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{3, 4},
                               {1.764052345967664f,
                                0.4001572083672233f,
                                0.9787379841057392f,
                                2.240893199201458f,
                                1.8675579901499675f,
                                -0.977277879876411f,
                                0.9500884175255894f,
                                -0.1513572082976979f,
                                -0.10321885179355784f,
                                0.41059850193837233f,
                                0.144043571160878f,
                                1.454273506962975f});
    test_case.add_expected_output<float>(Shape{3}, {5.3838407376420845f, 1.689011319501448f, 1.9056967282686674f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_float16_tensor_as_int32) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/conv_fp16_W_as_int32.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // clang-format off
    test_case.add_input<ngraph::float16>(Shape{1, 1, 4, 4},
            {   0,  1,  2,  3,
                4,  5,  6,  7,
                8,  9,  10, 11,
                12, 13, 14, 15  });
    /* filters
            [[[[0.25, 0.5, 0.25],
               [0.5,  1.0, 0.5],
               [0.25, 0.5, 0.25]]]] */
    test_case.add_expected_output<ngraph::float16>(Shape{1, 1, 2, 2},
            {   20, 24,
                36, 40  });
    // clang-format on
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_3d) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/max_pool_3d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{1, 3, 3}, {-1, 0, 1, 20, -20, 10, 0, 2, 1});
    test_case.add_expected_output<int32_t>(Shape{1, 3, 2}, {0, 1, 20, 10, 2, 2});
    test_case.add_expected_output<int64_t>(Shape{1, 3, 2}, {1, 2, 3, 5, 7, 7});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_ceil_mode) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/max_pool_4d_ceil_mode.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{1, 1, 4, 4}, gen_range<int32_t>(16, 1));
    test_case.add_expected_output<int32_t>(Shape{1, 1, 2, 2}, {11, 12, 15, 16});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {10, 11, 14, 15});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_dilations) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/max_pool_4d_dilations.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int32_t>(Shape{1, 1, 4, 4}, {9, 10, 11, 12, 1, 2, 3, 4, 16, 14, 15, 13, 5, 6, 8, 7});
    test_case.add_expected_output<int32_t>(Shape{1, 1, 2, 2}, {16, 14, 8, 7});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {8, 9, 14, 15});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_strides) {
    // kernel: 3x3
    // strides: 3, 3
    // explicit pads: 2, 2, 2, 2
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/max_pool_4d_strides.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int8_t>(Shape{1, 1, 5, 5}, gen_range<int8_t>(25, 1));
    test_case.add_expected_output<int8_t>(Shape{1, 1, 3, 3}, {1, 4, 5, 16, 19, 20, 21, 24, 25});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 3, 3}, {0, 3, 4, 15, 18, 19, 20, 23, 24});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_4d_ceil_strides) {
    // kernel: 3x3
    // strides: 2, 2
    // ceil_mode: 1
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/max_pool_4d_ceil_strides.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(
        Shape{1, 1, 4, 4},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    test_case.add_expected_output<float>(Shape{1, 1, 2, 2}, {11.0f, 12.0f, 15.0f, 16.0f});
    test_case.add_expected_output<int64_t>(Shape{1, 1, 2, 2}, {10, 11, 14, 15});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_uniform) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/random_uniform.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 2}, {43.45518f, 48.67585f, 42.227386f, 40.86294f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_uniform_like) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/random_uniform_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{2, 2}, {41, 42, 43, 44});
    test_case.add_expected_output<float>(Shape{2, 2}, {43.45518f, 48.67585f, 42.227386f, 40.86294f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_normal) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/random_normal.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<float>(Shape{2, 2}, {13.459274f, 41.75028f, -19.311913f, 131.79282f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_random_normal_like) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/random_normal_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{2, 2}, {0, 0, 0, 0});
    test_case.add_expected_output<float>(Shape{2, 2}, {13.459274f, 41.75028f, -19.311913f, 131.79282f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_2fin) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/aten_embedding_sum_packed_2in.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, -2.f, -2.2f, -0.19999999f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_3fin_offsets_none) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/aten_embedding_sum_packed_3in_offset_none.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, -2.f, -2.2f, -0.19999999f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_4fin_per_sample_weights) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/aten_embedding_sum_packed_4in_per_sample_weights.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});                  // indices
    test_case.add_input<float>(Shape{3, 2}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});  // per_sample_weights

    test_case.add_expected_output<float>(Shape{3, 2}, {-1.05f, -1.2f, -1.f, -1.1f, -0.09999999f, 0.4f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_packed_sum_4in_two_none) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/aten_embedding_sum_packed_4in_two_none.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{3, 2}, {0, 2, 1, 2, 3, 4});  // indices

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, -2.f, -2.2f, -0.19999999f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_offsets_sum_3in) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/aten_embedding_sum_offset_3in.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});  // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});     // offsets

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, 0.f, 0.f, -0.2f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_offsets_sum_4in) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/aten_embedding_sum_offset_4in.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});            // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});               // offsets
    test_case.add_input<float>(Shape{4}, {0.5f, 0.5f, 0.5f, 0.5f});  // per_sample_weights

    test_case.add_expected_output<float>(Shape{3, 2}, {-1.05f, -1.2f, 0.f, 0.f, -0.09999999f, 0.4f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_embedding_bag_many_node_outputs) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/aten_embedding_sum_many_outputs.onnx"));

    // 4 outputs in onnx Node (1 connected and 3 not connected)
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->get_results().size(), 1);

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{5, 2}, {-0.2f, -0.6f, -0.1f, -0.4f, -1.9f, -1.8f, -1.f, 1.5f, 0.8f, -0.7f});
    test_case.add_input<int32_t>(Shape{4}, {0, 2, 3, 4});  // indices
    test_case.add_input<int32_t>(Shape{3}, {0, 2, 2});     // offsets

    test_case.add_expected_output<float>(Shape{3, 2}, {-2.1f, -2.4f, 0.f, 0.f, -0.2f, 0.8f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_unsupported_embedding_mode) {
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/aten_unsupported_embedding_mode.onnx"));
        FAIL() << "Expected exception was not thrown.";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Unsupported mode, only `0` (sum) is supported as ATen embedding_bag `mode` attribute. Got: 1"));
    } catch (...) {
        FAIL() << "Other exception than expected was thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_aten_unsupported_operator) {
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/aten_unsupported_operator.onnx"));
        FAIL() << "Expected exception was not thrown.";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string(
                "Only `embedding_bag` is supported as ATen `operator` attribute. Got: test_unsupported_operator"));
    } catch (...) {
        FAIL() << "Other exception than expected was thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_ai_onnx_domain) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/unsqueeze_ai_onnx_domain.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_default_domain) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/unsqueeze_default_domain.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();

    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_default_domain_opset13) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/unsqueeze_default_domain_opset13.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();
    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unsqueeze_ai_onnx_domain_opset13) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/unsqueeze_ai_onnx_domain_opset13.onnx"));

    auto input = test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                          {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                     .get_vector();
    auto expected_output =
        test::NDArray<float, 4>({{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                                  {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_expand_failsafe_node) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/expand_failsafe_node.onnx"));

    auto test_case = test::TestCase(function, s_device);
    const auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    test_case.add_input<float>(input_data);
    // the target shape is an empty constant so the Expand operation should not modify the input shape
    test_case.add_expected_output<float>(input_data);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/scan15_fib_like.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>(10, 1));

    test_case.add_expected_output<float>(Shape{}, {55});
    test_case.add_expected_output<float>(Shape{}, {89});
    test_case.add_expected_output<float>(Shape{10}, {1., 2., 3., 5., 8., 13., 21., 34., 55., 89.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_out_rev) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/scan15_fib_like_out_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10}, std::vector<float>(10, 1));

    test_case.add_expected_output<float>(Shape{}, {55});
    test_case.add_expected_output<float>(Shape{}, {89});
    test_case.add_expected_output<float>(Shape{10}, {89., 55., 34., 21., 13., 8., 5., 3., 2., 1.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_input_rev) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/scan15_fib_like_input_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10},
                               std::vector<float>{0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});

    test_case.add_expected_output<float>(Shape{}, {0.14897026f});
    test_case.add_expected_output<float>(Shape{}, {0.f});
    test_case.add_expected_output<float>(
        Shape{10},
        {0.9f, 1.52f, 1.694f, 1.9284f, 1.8112f, 1.4958401f, 0.9921121f, 0.49759045f, 0.14897026f, 0.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_fib_like_input_out_rev) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/scan15_fib_like_input_out_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{}, {0});
    test_case.add_input<float>(Shape{}, {1});
    test_case.add_input<float>(Shape{10},
                               std::vector<float>{0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});

    test_case.add_expected_output<float>(Shape{}, {0.14897026f});
    test_case.add_expected_output<float>(Shape{}, {0.});
    test_case.add_expected_output<float>(
        Shape{10},
        {0.f, 0.14897026f, 0.49759045f, 0.9921121f, 1.4958401f, 1.8112f, 1.9284f, 1.694f, 1.52f, 0.9f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_ND_mixed_ones) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/scan15_ND_mixed.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0, 0, 0, 0, 0, 0});
    test_case.add_input<float>(Shape{1, 3, 2}, {1, 1, 1, 1, 1, 1});
    test_case.add_input<float>(Shape{1, 3, 5, 2}, std::vector<float>(30, 1));  // multiply by one
    test_case.add_input<float>(Shape{1, 5, 3, 2}, std::vector<float>(30, 1));  // div by one

    test_case.add_expected_output<float>(Shape{1, 3, 2}, {5., 5., 5., 5., 5., 5.});
    test_case.add_expected_output<float>(Shape{1, 3, 2}, {8., 8., 8., 8., 8., 8.});
    test_case.add_expected_output<float>(Shape{1, 3, 2, 5},
                                         {8., 5., 3., 2., 1., 8., 5., 3., 2., 1., 8., 5., 3., 2., 1.,
                                          8., 5., 3., 2., 1., 8., 5., 3., 2., 1., 8., 5., 3., 2., 1.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15f_ND_mixed_vals) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/scan15_ND_mixed.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_input<float>(Shape{1, 3, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    std::vector<float> sequence_vals{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f,
                                     1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f,
                                     2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938f, 2.1428573f, 21.070545f, 16.92727f, 49.765778f, 41.444443f});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943f, 0.5274726f, 16.80789f, 14.025973f, 59.98805f, 50.518517f});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943f, 2.7327938f, 7.3076925f, 10.f,       9.f,        0.5274726f, 2.1428573f, 4.714286f,
         6.f,         5.f,        16.80789f,  21.070545f, 20.185184f, 13.851851f, 6.333333f,  14.025973f,
         16.92727f,   15.799998f, 10.799999f, 5.f,        59.98805f,  49.765778f, 33.074867f, 16.690908f,
         5.8f,        50.518517f, 41.444443f, 27.444445f, 14.f,       5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_ND_mixed_vals_neg_axes) {
    // Negative indices for scan_input_axes and scan_output_axes attributes
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/scan15_ND_mixed_neg_axes.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_input<float>(Shape{1, 3, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    std::vector<float> sequence_vals{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f,
                                     1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f,
                                     2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938f, 2.1428573f, 21.070545f, 16.92727f, 49.765778f, 41.444443f});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943f, 0.5274726f, 16.80789f, 14.025973f, 59.98805f, 50.518517f});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943f, 2.7327938f, 7.3076925f, 10.f,       9.f,        0.5274726f, 2.1428573f, 4.714286f,
         6.f,         5.f,        16.80789f,  21.070545f, 20.185184f, 13.851851f, 6.333333f,  14.025973f,
         16.92727f,   15.799998f, 10.799999f, 5.f,        59.98805f,  49.765778f, 33.074867f, 16.690908f,
         5.8f,        50.518517f, 41.444443f, 27.444445f, 14.f,       5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_dyn_rank_vals) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/scan15_dyn_rank.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 3, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    test_case.add_input<float>(Shape{1, 3, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    std::vector<float> sequence_vals{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f,
                                     1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f,
                                     2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.f};
    test_case.add_input<float>(Shape{1, 3, 5, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{1, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {2.7327938f, 2.1428573f, 21.070545f, 16.92727f, 49.765778f, 41.444443f});
    test_case.add_expected_output<float>(Shape{1, 3, 2},
                                         {0.40161943f, 0.5274726f, 16.80789f, 14.025973f, 59.98805f, 50.518517f});
    test_case.add_expected_output<float>(
        Shape{1, 3, 2, 5},
        {0.40161943f, 2.7327938f, 7.3076925f, 10.f,       9.f,        0.5274726f, 2.1428573f, 4.714286f,
         6.f,         5.f,        16.80789f,  21.070545f, 20.185184f, 13.851851f, 6.333333f,  14.025973f,
         16.92727f,   15.799998f, 10.799999f, 5.f,        59.98805f,  49.765778f, 33.074867f, 16.690908f,
         5.8f,        50.518517f, 41.444443f, 27.444445f, 14.f,       5.f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_dyn_rank_vals_neg_axes) {
    // Negative indices for scan_input_axes and scan_output_axes attributes
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/scan15_dyn_rank_neg_axes.onnx"));
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Rank must be static in order to normalize negative axis"));
    } catch (...) {
        FAIL() << "Expected exception was not thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan15_ND_b4_input_rev_vals) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/scan15_ND_b4_input_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0.f));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1.f));
    std::vector<float> sequence_vals{
        0.1f,  0.2f,  0.3f,  0.4f,  0.5f, 0.6f,  0.7f,  0.8f,  0.9f,  1.f,   1.1f,  1.2f,  1.3f,  1.4f,  1.5f,
        1.6f,  1.7f,  1.8f,  1.9f,  2.f,  2.1f,  2.2f,  2.3f,  2.4f,  2.5f,  2.6f,  2.7f,  2.8f,  2.9f,  3.f,
        3.1f,  3.2f,  3.3f,  3.4f,  3.5f, 3.6f,  3.7f,  3.8f,  3.9f,  4.f,   4.1f,  4.2f,  4.3f,  4.4f,  4.5f,
        4.6f,  4.7f,  4.8f,  4.9f,  5.f,  5.1f,  5.2f,  5.3f,  5.4f,  5.5f,  5.6f,  5.7f,  5.8f,  5.9f,  6.f,
        6.1f,  6.2f,  6.3f,  6.4f,  6.5f, 6.6f,  6.7f,  6.8f,  6.9f,  7.f,   7.1f,  7.2f,  7.3f,  7.4f,  7.5f,
        7.6f,  7.7f,  7.8f,  7.9f,  8.f,  8.1f,  8.2f,  8.3f,  8.4f,  8.5f,  8.6f,  8.7f,  8.8f,  8.9f,  9.f,
        9.1f,  9.2f,  9.3f,  9.4f,  9.5f, 9.6f,  9.7f,  9.8f,  9.9f,  10.f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f,
        10.6f, 10.7f, 10.8f, 10.9f, 11.f, 11.1f, 11.2f, 11.3f, 11.4f, 11.5f, 11.6f, 11.7f, 11.8f, 11.9f, 12.f};
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply factor (areverse)
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.8880844f, 6.83f,
         6.7754016f, 6.7239814f, 6.6754713f, 6.6296296f, 5.9686656f, 5.953226f,  5.9382715f, 5.9237804f,
         5.9097314f, 5.896105f,  5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f});
    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {6.271278f, 6.2461543f, 6.2433867f, 6.2545457f, 6.2744985f, 6.3f,       6.9531364f, 6.970527f,
         6.987378f, 7.003712f,  7.019554f,  7.034921f,  7.30868f,   7.3164845f, 7.324116f,  7.3315806f,
         7.338885f, 7.346032f,  7.485426f,  7.489783f,  7.494067f,  7.49828f,   7.5024257f, 7.506502f});
    test_case.add_expected_output<float>(
        Shape{5, 4, 3, 2},
        {25.f,       13.f,       9.f,        7.f,        5.8f,       5.f,        1.7741936f, 1.75f,      1.7272727f,
         1.7058823f, 1.6857144f, 1.6666667f, 1.3934426f, 1.3870969f, 1.3809522f, 1.375f,     1.3692307f, 1.3636364f,
         1.2637362f, 1.2608696f, 1.2580644f, 1.2553192f, 1.2526315f, 1.25f,      70.57143f,  35.f,       23.333334f,
         17.6f,      14.218181f, 12.f,       3.6739323f, 3.618421f,  3.5664334f, 3.5176468f, 3.471777f,  3.4285717f,
         2.822119f,  2.8083491f, 2.7950313f, 2.7821426f, 2.7696643f, 2.757576f,  2.543786f,  2.5377107f, 2.5317693f,
         2.5259573f, 2.520271f,  2.514706f,  95.57143f,  47.999996f, 32.333336f, 24.6f,      20.01818f,  17.f,
         5.448126f,  5.368421f,  5.293706f,  5.223529f,  5.157491f,  5.0952387f, 4.215562f,  4.195446f,  4.1759834f,
         4.1571426f, 4.138895f,  4.1212125f, 3.8075223f, 3.7985802f, 3.7898335f, 3.7812767f, 3.7729027f, 3.764706f,
         61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.8880844f, 6.83f,      6.7754016f,
         6.7239814f, 6.6754713f, 6.6296296f, 5.9686656f, 5.953226f,  5.9382715f, 5.9237804f, 5.9097314f, 5.896105f,
         5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f,  6.271278f,  6.2461543f, 6.2433867f,
         6.2545457f, 6.2744985f, 6.3f,       6.9531364f, 6.970527f,  6.987378f,  7.003712f,  7.019554f,  7.034921f,
         7.30868f,   7.3164845f, 7.324116f,  7.3315806f, 7.338885f,  7.346032f,  7.485426f,  7.489783f,  7.494067f,
         7.49828f,   7.5024257f, 7.506502f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_ones) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/scan8_ND_b4.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1));
    std::vector<float> sequence_vals(120, 1);
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply by one
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div by one

    test_case.add_expected_output<float>(Shape{4, 3, 2}, {5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                                                          5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.});
    test_case.add_expected_output<float>(Shape{4, 3, 2}, {8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
                                                          8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.});
    test_case.add_expected_output<float>(
        Shape{4, 5, 3, 2},
        {1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 5., 5., 5., 5., 5., 5.,
         8., 8., 8., 8., 8., 8., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3.,
         5., 5., 5., 5., 5., 5., 8., 8., 8., 8., 8., 8., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2.,
         3., 3., 3., 3., 3., 3., 5., 5., 5., 5., 5., 5., 8., 8., 8., 8., 8., 8., 1., 1., 1., 1., 1., 1.,
         2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 5., 5., 5., 5., 5., 5., 8., 8., 8., 8., 8., 8.});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_input_rev_vals) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/scan8_ND_b4_input_rev.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 0.f));
    test_case.add_input<float>(Shape{4, 3, 2}, std::vector<float>(24, 1.f));
    std::vector<float> sequence_vals{
        0.1f,  0.2f,  0.3f,  0.4f,  0.5f, 0.6f,  0.7f,  0.8f,  0.9f,  1.f,   1.1f,  1.2f,  1.3f,  1.4f,  1.5f,
        1.6f,  1.7f,  1.8f,  1.9f,  2.f,  2.1f,  2.2f,  2.3f,  2.4f,  2.5f,  2.6f,  2.7f,  2.8f,  2.9f,  3.f,
        3.1f,  3.2f,  3.3f,  3.4f,  3.5f, 3.6f,  3.7f,  3.8f,  3.9f,  4.f,   4.1f,  4.2f,  4.3f,  4.4f,  4.5f,
        4.6f,  4.7f,  4.8f,  4.9f,  5.f,  5.1f,  5.2f,  5.3f,  5.4f,  5.5f,  5.6f,  5.7f,  5.8f,  5.9f,  6.f,
        6.1f,  6.2f,  6.3f,  6.4f,  6.5f, 6.6f,  6.7f,  6.8f,  6.9f,  7.f,   7.1f,  7.2f,  7.3f,  7.4f,  7.5f,
        7.6f,  7.7f,  7.8f,  7.9f,  8.f,  8.1f,  8.2f,  8.3f,  8.4f,  8.5f,  8.6f,  8.7f,  8.8f,  8.9f,  9.f,
        9.1f,  9.2f,  9.3f,  9.4f,  9.5f, 9.6f,  9.7f,  9.8f,  9.9f,  10.f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f,
        10.6f, 10.7f, 10.8f, 10.9f, 11.f, 11.1f, 11.2f, 11.3f, 11.4f, 11.5f, 11.6f, 11.7f, 11.8f, 11.9f, 12.f};
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // multiply factor (reverse)
    test_case.add_input<float>(Shape{4, 5, 3, 2}, sequence_vals);  // div factor

    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.8880844f, 6.83f,
         6.7754016f, 6.7239814f, 6.6754713f, 6.6296296f, 5.9686656f, 5.953226f,  5.9382715f, 5.9237804f,
         5.9097314f, 5.896105f,  5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f});
    test_case.add_expected_output<float>(
        Shape{4, 3, 2},
        {6.271278f, 6.2461543f, 6.2433867f, 6.2545457f, 6.2744985f, 6.3f,       6.9531364f, 6.970527f,
         6.987378f, 7.003712f,  7.019554f,  7.034921f,  7.30868f,   7.3164845f, 7.324116f,  7.3315806f,
         7.338885f, 7.346032f,  7.485426f,  7.489783f,  7.494067f,  7.49828f,   7.5024257f, 7.506502f});
    test_case.add_expected_output<float>(
        Shape{4, 5, 3, 2},
        {25.f,       13.f,       9.f,        7.f,        5.8f,       5.f,        70.57143f,  35.f,       23.333334f,
         17.6f,      14.218181f, 12.f,       95.57143f,  47.999996f, 32.333336f, 24.6f,      20.01818f,  17.f,
         61.210526f, 33.2f,      23.857145f, 19.181818f, 16.373913f, 14.5f,      6.271278f,  6.2461543f, 6.2433867f,
         6.2545457f, 6.2744985f, 6.3f,       1.7741936f, 1.75f,      1.7272727f, 1.7058823f, 1.6857144f, 1.6666667f,
         3.6739323f, 3.618421f,  3.5664334f, 3.5176468f, 3.471777f,  3.4285717f, 5.448126f,  5.368421f,  5.293706f,
         5.223529f,  5.157491f,  5.0952387f, 6.8880844f, 6.83f,      6.7754016f, 6.7239814f, 6.6754713f, 6.6296296f,
         6.9531364f, 6.970527f,  6.987378f,  7.003712f,  7.019554f,  7.034921f,  1.3934426f, 1.3870969f, 1.3809522f,
         1.375f,     1.3692307f, 1.3636364f, 2.822119f,  2.8083491f, 2.7950313f, 2.7821426f, 2.7696643f, 2.757576f,
         4.215562f,  4.195446f,  4.1759834f, 4.1571426f, 4.138895f,  4.1212125f, 5.9686656f, 5.953226f,  5.9382715f,
         5.9237804f, 5.9097314f, 5.896105f,  7.30868f,   7.3164845f, 7.324116f,  7.3315806f, 7.338885f,  7.346032f,
         1.2637362f, 1.2608696f, 1.2580644f, 1.2553192f, 1.2526315f, 1.25f,      2.543786f,  2.5377107f, 2.5317693f,
         2.5259573f, 2.520271f,  2.514706f,  3.8075223f, 3.7985802f, 3.7898335f, 3.7812767f, 3.7729027f, 3.764706f,
         5.652082f,  5.645059f,  5.638186f,  5.6314588f, 5.624872f,  5.618421f,  7.485426f,  7.489783f,  7.494067f,
         7.49828f,   7.5024257f, 7.506502f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_scan8_ND_b4_seq_lens) {
    // ONNX Scan-8 can has optional `sequence_lens` input, the input was removed since ONNX Scan-9
    try {
        const auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/scan8_ND_b4_seq_lens.onnx"));
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string(" ONNX Scan-8 `sequence_lens` input is not supported. "));
    } catch (...) {
        FAIL() << "Expected exception was not thrown.";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_softsign) {
    auto model = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/softsign.onnx"));

    Inputs inputs{std::vector<float>{1.0f, 0.1f, 20.0f, 12.0f, -12.0f, -0.2f, 0.5f, 100.0f, 0.0f, -1.0f}};

    std::vector<float> output{0.5f,
                              0.09090909f,
                              0.95238096f,
                              0.9230769f,
                              -0.9230769f,
                              -0.16666666f,
                              0.33333334f,
                              0.990099f,
                              0.f,
                              -0.5f};

    auto test_case = test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_grid_sample) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/grid_sample.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>(Shape{1, 1, 4, 4}, gen_range<float>(16));
    test_case.add_input<float>(
        Shape{1, 6, 6, 2},
        {-1.0000f, -1.0000f, -0.6000f, -1.0000f, -0.2000f, -1.0000f, 0.2000f,  -1.0000f, 0.6000f,  -1.0000f, 1.0000f,
         -1.0000f, -1.0000f, -0.6000f, -0.6000f, -0.6000f, -0.2000f, -0.6000f, 0.2000f,  -0.6000f, 0.6000f,  -0.6000f,
         1.0000f,  -0.6000f, -1.0000f, -0.2000f, -0.6000f, -0.2000f, -0.2000f, -0.2000f, 0.2000f,  -0.2000f, 0.6000f,
         -0.2000f, 1.0000f,  -0.2000f, -1.0000f, 0.2000f,  -0.6000f, 0.2000f,  -0.2000f, 0.2000f,  0.2000f,  0.2000f,
         0.6000f,  0.2000f,  1.0000f,  0.2000f,  -1.0000f, 0.6000f,  -0.6000f, 0.6000f,  -0.2000f, 0.6000f,  0.2000f,
         0.6000f,  0.6000f,  0.6000f,  1.0000f,  0.6000f,  -1.0000f, 1.0000f,  -0.6000f, 1.0000f,  -0.2000f, 1.0000f,
         0.2000f,  1.0000f,  0.6000f,  1.0000f,  1.0000f,  1.0000});

    test_case.add_expected_output<float>(
        Shape{1, 1, 6, 6},
        {0.0000f,  0.1500f,  0.5500f, 0.9500f, 1.3500f,  0.7500f, 0.6000f, 1.5000f,  2.3000f,
         3.1000f,  3.9000f,  2.1000f, 2.2000f, 4.7000f,  5.5000f, 6.3000f, 7.1000f,  3.7000f,
         3.8000f,  7.9000f,  8.7000f, 9.5000f, 10.3000f, 5.3000f, 5.4000f, 11.1000f, 11.9000f,
         12.7000f, 13.5000f, 6.9000f, 3.0000f, 6.1500f,  6.5500f, 6.9500f, 7.3500f,  3.7500});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_concat_empty_init) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/concat_empty_init.onnx"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<int64_t>(Shape{2}, std::vector<int64_t>{1, 2});
    test_case.add_expected_output<int64_t>(Shape{2}, std::vector<int64_t>{1, 2});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_basic) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/trilu_basic.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // clang-format off
    test_case.add_input<float>(Shape{5, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25});
    test_case.add_expected_output<float>(Shape{5, 5},
        std::vector<float>{ 1,  0,  0,  0,  0,
                            6,  7,  0,  0,  0,
                           11, 12, 13,  0,  0,
                           16, 17, 18, 19,  0,
                           21, 22, 23, 24, 25});
    // clang-format on
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_lower) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/trilu_lower.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // clang-format off
    test_case.add_input<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{4, 5},
        std::vector<float>{ 1,  0,  0,  0,  0,
                            6,  7,  0,  0,  0,
                           11, 12, 13,  0,  0,
                           16, 17, 18, 19,  0});
    test_case.run();

    test_case.add_input<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {2}); // k
    test_case.add_expected_output<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  0,  0,
                            6,  7,  8,  9,  0,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.run();

    test_case.add_input<float>(Shape{4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {-2}); // k
    test_case.add_expected_output<float>(Shape{4, 5},
        std::vector<float>{ 0,  0,  0,  0,  0,
                            0,  0,  0,  0,  0,
                           11,  0,  0,  0,  0,
                           16, 17,  0,  0,  0});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_upper) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/trilu_upper.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // clang-format off

    test_case.add_input<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            0,  6,  7,  8,
                            0,  0, 11, 12,
                            0,  0,  0, 16,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {1}); // k
    test_case.add_expected_output<float>(Shape{5, 4},
        std::vector<float>{ 0,  2,  3,  4,
                            0,  0,  7,  8,
                            0,  0,  0, 12,
                            0,  0,  0,  0,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20});
    test_case.add_input<int64_t>(Shape{}, {-1}); // k
    test_case.add_expected_output<float>(Shape{5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            0, 10, 11, 12,
                            0,  0, 15, 16,
                            0,  0,  0, 20});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_upper_3d) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/trilu_upper_3d.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // clang-format off

    test_case.add_input<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                           33, 34, 35, 36,
                           37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            0,  6,  7,  8,
                            0,  0, 11, 12,
                            0,  0,  0, 16,
                            0,  0,  0,  0,

                           21, 22, 23, 24,
                            0, 26, 27, 28,
                            0,  0, 31, 32,
                            0,  0,  0, 36,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                           33, 34, 35, 36,
                           37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {2}); // k
    test_case.add_expected_output<float>(Shape{2, 5, 4},
        std::vector<float>{ 0,  0,  3,  4,
                            0,  0,  0,  8,
                            0,  0,  0,  0,
                            0,  0,  0,  0,
                            0,  0,  0,  0,

                            0,  0, 23, 24,
                            0,  0,  0, 28,
                            0,  0,  0,  0,
                            0,  0,  0,  0,
                            0,  0,  0,  0});
    test_case.run();

    test_case.add_input<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                           13, 14, 15, 16,
                           17, 18, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                           33, 34, 35, 36,
                           37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {-2}); // k
    test_case.add_expected_output<float>(Shape{2, 5, 4},
        std::vector<float>{ 1,  2,  3,  4,
                            5,  6,  7,  8,
                            9, 10, 11, 12,
                            0, 14, 15, 16,
                            0,  0, 19, 20,

                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32,
                            0, 34, 35, 36,
                            0,  0, 39, 40});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_lower_4d) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/trilu_lower_4d.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {0}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  0,  0,  0,  0,
                            6,  7,  0,  0,  0,
                           11, 12, 13,  0,  0,
                           16, 17, 18, 19,  0,

                           21,  0,  0,  0,  0,
                           26, 27,  0,  0,  0,
                           31, 32, 33,  0,  0,
                           36, 37, 38, 39,  0});
    test_case.run();

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {1}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  0,  0,  0,
                            6,  7,  8,  0,  0,
                           11, 12, 13, 14,  0,
                           16, 17, 18, 19, 20,

                           21, 22,  0,  0,  0,
                           26, 27, 28,  0,  0,
                           31, 32, 33, 34,  0,
                           36, 37, 38, 39, 40});
    test_case.run();

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {-1}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 0,  0,  0,  0,  0,
                            6,  0,  0,  0,  0,
                           11, 12,  0,  0,  0,
                           16, 17, 18,  0,  0,

                            0,  0,  0,  0,  0,
                           26,  0,  0,  0,  0,
                           31, 32,  0,  0,  0,
                           36, 37, 38,  0,  0});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_trilu_dynamic_shapes) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/dynamic_shapes/trilu_lower.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  3,  4,  5,
                            6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20,

                           21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40});
    test_case.add_input<int64_t>(Shape{}, {1}); // k
    test_case.add_expected_output<float>(Shape{2, 1, 4, 5},
        std::vector<float>{ 1,  2,  0,  0,  0,
                            6,  7,  8,  0,  0,
                           11, 12, 13, 14,  0,
                           16, 17, 18, 19, 20,

                           21, 22,  0,  0,  0,
                           26, 27, 28,  0,  0,
                           31, 32, 33, 34,  0,
                           36, 37, 38, 39, 40});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_finite) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/is_finite.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{1, 2, 3}, {std::nanf(""), std::numeric_limits<float>::infinity(), -0.6000f, -1.0000f, std::nanf(""), -1.0000f});

    test_case.add_expected_output<bool>(
        Shape{1, 2, 3},
        {false, false, true, true, false, true});

    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_default) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/is_inf.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{true, false,
                          false, false,
                          true, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_negative_only) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/is_inf_negative.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{false, false,
                          false, false,
                          true, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_positive_only) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/is_inf_positive.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{true, false,
                          false, false,
                          false, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_inf_detect_none) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/is_inf_none.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(
        Shape{2, 2, 2},
        std::vector<float>{ std::numeric_limits<float>::infinity(), 0.0000f,
                            std::numeric_limits<float>::max(), -0.5000f,
                            -std::numeric_limits<float>::infinity(), 1.0000f,
                            std::numeric_limits<float>::min(), std::nanf("")});
    test_case.add_expected_output<bool>(
        Shape{2, 2, 2},
        std::vector<bool>{false, false,
                          false, false,
                          false, false,
                          false, false});
    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_is_nan) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/is_nan.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // clang-format off

    test_case.add_input<float>(Shape{1, 2, 3}, {std::nanf(""), std::nanf(""), -0.6000f, -1.0000f, std::nanf(""), -1.0000f});

    test_case.add_expected_output<bool>(
        Shape{1, 2, 3},
        {true, true, false, false, true, false});

    test_case.run();

    // clang-format on
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_squeeze_default_domain_opset13) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/squeeze_default_domain_opset13.onnx"));

    auto input =
        test::NDArray<float, 3>({{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}).get_vector();
    auto expected_output =
        test::NDArray<float, 2>({{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}).get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_of_shape_empty_init) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_of_shape_empty_init.onnx"));
    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int32_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_constant_of_shape_null_node) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/constant_of_shape_null_node.onnx"));
    auto test_case = test::TestCase(function, s_device);
    test_case.add_expected_output<int32_t>(Shape{}, {1});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float16_to_uint32) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_float16_to_uint32.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<ngraph::float16>(Shape{1, 1, 2, 2}, std::vector<ngraph::float16>{1.5f, 2.3f, 3.f, 4.f});
    test_case.add_input<uint32_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<uint32_t>(std::vector<uint32_t>{1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float16_to_int64) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_float16_to_int64.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<ngraph::float16>(Shape{1, 1, 2, 2}, std::vector<ngraph::float16>{1.5f, 2.3f, 3.f, 4.f});
    test_case.add_input<int64_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(std::vector<int64_t>{1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, DISABLED_castlike_int8_to_uint16) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_int8_to_uint16.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int8_t>(Shape{1, 1, 2, 2}, std::vector<int8_t>{-1, -2, 3, 4});
    test_case.add_input<uint16_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<uint16_t>(std::vector<uint16_t>{65535, 65534, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float64_to_int64) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_float64_to_int64.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<double>(Shape{1, 1, 2, 2}, std::vector<double>{1.5, 2.3, 3, 4});
    test_case.add_input<int64_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(std::vector<int64_t>{1, 2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_int8_to_float16) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_int8_to_float16.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int8_t>(Shape{1, 1, 2, 2}, std::vector<int8_t>{-127, -2, 3, 4});
    test_case.add_input<ngraph::float16>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<ngraph::float16>(std::vector<ngraph::float16>{-127.0, -2.0, 3.0, 4.0});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_int32_to_float) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_int32_to_float64.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>(Shape{1, 1, 2, 2}, std::vector<int32_t>{-1, 2, 3, 4});
    test_case.add_input<float>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<float>(std::vector<float>{-1.0f, 2.0f, 3.0f, 4.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, castlike_float64_to_int32) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_float64_to_int32.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(Shape{1, 1, 2, 2}, std::vector<float>{-107374.9876543f, -2.2f, 3.3f, 4.4f});
    test_case.add_input<int32_t>(Shape{4}, {1, 2, 3, 4});
    test_case.add_expected_output<int32_t>(std::vector<int32_t>{-107374, -2, 3, 4});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, DISABLED_castlike_float32_to_bfloat16) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_float32_to_bfloat16.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>(
        Shape{3, 4},
        std::vector<float>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.8f, 9.f, 10.f, 11.f, 12.f});
    test_case.add_input<bfloat16>(Shape{3, 4},
                                  {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f});
    test_case.add_expected_output<bfloat16>(
        std::vector<bfloat16>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.8f, 9.f, 10.f, 11.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, DISABLED_castlike_bfloat16_to_float32) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/castlike_bfloat16_to_float32.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<bfloat16>(
        Shape{3, 4},
        std::vector<bfloat16>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.8f, 9.f, 10.f, 11.f, 12.f});
    test_case.add_input<float>(Shape{3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    test_case.add_expected_output<float>(
        std::vector<float>{121.5f, 122.7f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.75f, 9.f, 10.f, 11.f, 12.f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_3d_default_attributes) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/unique_3d_default_attributes.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({9, 12, 3, 121, 5, 4, 10, 9});
    test_case.add_expected_output<int32_t>(Shape{7}, {3, 4, 5, 9, 10, 12, 121});
    test_case.add_expected_output<int64_t>(Shape{7}, {2, 5, 4, 0, 6, 1, 3});
    test_case.add_expected_output<int64_t>(Shape{8}, {3, 5, 0, 6, 2, 1, 4, 3});
    test_case.add_expected_output<int64_t>(Shape{7}, {1, 1, 1, 2, 1, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_1d_no_duplicates) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/unique_1d_no_duplicates.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({5, 4, 3, 2, 1});
    test_case.add_expected_output<int32_t>(Shape{5}, {5, 4, 3, 2, 1});
    test_case.add_expected_output<int64_t>(Shape{5}, {0, 1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(Shape{5}, {0, 1, 2, 3, 4});
    test_case.add_expected_output<int64_t>(Shape{5}, {1, 1, 1, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_1d_no_duplicates_sorted) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/unique_1d_no_duplicates_sorted.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({5, 4, 3, 2, 1});
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_expected_output<int64_t>(Shape{5}, {4, 3, 2, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{5}, {4, 3, 2, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{5}, {1, 1, 1, 1, 1});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_3d_with_duplicates_and_axis) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/unique_3d_with_duplicates_and_axis.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    test_case.add_expected_output<int32_t>(Shape{1, 2, 3}, {1, 2, 3, 4, 5, 6});
    test_case.add_expected_output<int64_t>(Shape{1}, {0});
    test_case.add_expected_output<int64_t>(Shape{2}, {0, 0});
    test_case.add_expected_output<int64_t>(Shape{1}, {2});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_unique_3d_with_duplicates_and_axis_2) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/unique_3d_with_duplicates_and_axis_2.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<int32_t>({-1, 2, -1, 5, -3, 5, 7, -8, 7, 4, 4, 4});
    test_case.add_expected_output<int32_t>(Shape{2, 2, 2}, {-1, 2, 5, -3, 7, -8, 4, 4});
    test_case.add_expected_output<int64_t>(Shape{2}, {0, 1});
    test_case.add_expected_output<int64_t>(Shape{3}, {0, 1, 0});
    test_case.add_expected_output<int64_t>(Shape{2}, {2, 1});

    test_case.run();
}
