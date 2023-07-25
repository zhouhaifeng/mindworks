// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "library_extension.hpp"

#include <common_test_utils/graph_comparator.hpp>
#include <ostream>

#include "common_test_utils/file_utils.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/swish.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontendLibraryExtensionTest::getTestCaseName(
    const testing::TestParamInfo<FrontendLibraryExtensionTestParams>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontendLibraryExtensionTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontendLibraryExtensionTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

inline std::string get_lib_path(const std::string& lib_name) {
    return ov::util::make_plugin_library_name<char>(CommonTestUtils::getExecutableDirectory(),
                                                    lib_name + IE_BUILD_POSTFIX);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontendLibraryExtensionTest, verifyFunctions) {
    std::shared_ptr<ov::Model> function_ref;
    {
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);

        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelName));
        ASSERT_NE(m_inputModel, nullptr);

        ASSERT_NO_THROW(function_ref = m_frontEnd->convert(m_inputModel));
        ASSERT_NE(function_ref, nullptr);

        const auto nodes = function_ref->get_ops();
        ASSERT_NE(std::find_if(nodes.begin(),
                               nodes.end(),
                               [](const std::shared_ptr<ov::Node>& n) {
                                   return ov::is_type<ov::op::v0::Relu>(n);
                               }),
                  nodes.end());
    }

    std::shared_ptr<ov::Model> function;
    {
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);

        const auto& lib_path = get_lib_path("test_builtin_extensions");
        m_frontEnd->add_extension(lib_path);

        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelName));
        ASSERT_NE(m_inputModel, nullptr);

        ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
        ASSERT_NE(function, nullptr);

        const auto nodes = function->get_ops();
        ASSERT_EQ(std::find_if(nodes.begin(),
                               nodes.end(),
                               [](const std::shared_ptr<ov::Node>& n) {
                                   return ov::is_type<ov::op::v0::Relu>(n);
                               }),
                  nodes.end());
        ASSERT_NE(std::find_if(nodes.begin(),
                               nodes.end(),
                               [](const std::shared_ptr<ov::Node>& n) {
                                   return ov::is_type<ov::op::v4::Swish>(n);
                               }),
                  nodes.end());
    }
}
