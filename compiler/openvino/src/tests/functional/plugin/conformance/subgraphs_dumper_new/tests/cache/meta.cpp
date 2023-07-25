// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "gtest/gtest.h"
#include "pugixml.hpp"

#include "openvino/openvino.hpp"
#include "openvino/util/file_util.hpp"

#include "common_test_utils/file_utils.hpp"

#include "cache/meta/meta_info.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ======================== Input Info Unit tests =============================================

class InputInfoUnitTest : public ::testing::Test {};

TEST_F(InputInfoUnitTest, constructor) {
    ASSERT_NO_THROW(auto in_info = InputInfo());
    ASSERT_NO_THROW(auto in_info = InputInfo(0));
    ASSERT_NO_THROW(auto in_info = InputInfo(0, 1));
    ASSERT_NO_THROW(auto in_info = InputInfo(0, 1, true));
}

TEST_F(InputInfoUnitTest, update_ranges) {
    auto in_info_0 = InputInfo();
    auto in_info_1 = InputInfo(0);
    in_info_0 = in_info_1;
    ASSERT_EQ(in_info_0.ranges.min, in_info_1.ranges.min);
    ASSERT_EQ(in_info_0.ranges.max, in_info_1.ranges.max);
    ASSERT_EQ(in_info_0.is_const, in_info_1.is_const);

    auto in_info_2 = InputInfo(1, 2);
    auto ref_in_info = InputInfo(0, 2);
    in_info_0 = in_info_2;
    ASSERT_EQ(in_info_0.ranges.min, ref_in_info.ranges.min);
    ASSERT_EQ(in_info_0.ranges.max, ref_in_info.ranges.max);
    ASSERT_EQ(in_info_0.is_const, ref_in_info.is_const);
}

// ======================== Model Info Func tests =============================================

class ModelInfoFuncTest : public ::testing::Test {};

TEST_F(ModelInfoFuncTest, constructor) {
    ASSERT_NO_THROW(auto model_info = ModelInfo());
    ASSERT_NO_THROW(auto model_info = ModelInfo("model.xml"));
    ASSERT_NO_THROW(auto model_info = ModelInfo("model.xml", 1));
    ASSERT_NO_THROW(auto model_info = ModelInfo("model.xml", 1, 2));
}

// ======================== Meta Info Functional tests =============================================

class MetaInfoFuncTest : public ::testing::Test {
protected:
    std::string test_model_path, test_model_name;
    std::map<std::string, InputInfo> test_in_info;
    std::map<std::string, ModelInfo> test_model_info;
    std::string test_artifacts_dir;

    void SetUp() override {
        test_model_path = "test_model_path.xml";
        test_model_name = CommonTestUtils::replaceExt(test_model_path, "");
        test_in_info = {{ "test_in_0", InputInfo(DEFAULT_MIN_VALUE, 1, true) }};
        test_model_info = {{ test_model_name, ModelInfo(test_model_path, 5) }};
        test_artifacts_dir = ov::util::path_join({CommonTestUtils::getCurrentWorkingDir(), "test_artifacts"});
        ov::util::create_directory_recursive(test_artifacts_dir);
    }

    void TearDown() override {
        CommonTestUtils::removeDir(test_artifacts_dir);
    }
};

TEST_F(MetaInfoFuncTest, constructor) {
    ASSERT_NO_THROW(auto meta = MetaInfo());
    ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name));
    ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name, test_in_info));
    ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name, test_in_info, 2));
    ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name, test_in_info, 3));
}

TEST_F(MetaInfoFuncTest, get_input_info) {
    auto test_meta = MetaInfo(test_model_name, test_in_info);
    ASSERT_NO_THROW(test_meta.get_input_info());
    ASSERT_EQ(test_meta.get_input_info(), test_in_info);
}

TEST_F(MetaInfoFuncTest, get_model_info) {
    auto test_meta = MetaInfo(test_model_path, test_in_info, 5);
    ASSERT_NO_THROW(test_meta.get_model_info());
    ASSERT_EQ(test_meta.get_model_info(), test_model_info);
}

TEST_F(MetaInfoFuncTest, update) {
    std::map<std::string, InputInfo> test_in_info = {{ "test_in_0", InputInfo(DEFAULT_MIN_VALUE, 1, true) }};
    auto test_meta = MetaInfo(test_model_name, test_in_info);
    std::map<std::string, InputInfo> test_input_info_1 = {{ "test_in_0", InputInfo(0, 1, true) }};
    std::string test_model_1 = "test_model_1";
    std::string test_model_path_1 = ov::util::path_join({ "path", "to",  test_model_1 + ".xml"});
    ASSERT_ANY_THROW(test_meta.update(test_model_path_1, {}));
    ASSERT_ANY_THROW(test_meta.update(test_model_path_1, {{ "test_in_1", InputInfo() }}));
    ASSERT_ANY_THROW(test_meta.update(test_model_path_1, {{ "test_in_0", InputInfo(0, 1, false) }}));
    ASSERT_NO_THROW(test_meta.update(test_model_path_1, test_input_info_1));
    ASSERT_NO_THROW(test_meta.update(test_model_path_1, test_input_info_1, 2));
}

TEST_F(MetaInfoFuncTest, serialize) {
    auto test_meta = MetaInfo(test_model_name, test_in_info);
    std::string seriliazation_path(ov::util::path_join({test_artifacts_dir, "test_meta.meta"}));
    test_meta.serialize(seriliazation_path);
    ASSERT_TRUE(ov::util::file_exists(seriliazation_path));
}

// ======================== Meta Info Unit tests =============================================

class MetaInfoUnitTest : public MetaInfoFuncTest,
                         public virtual MetaInfo {
protected:
    void SetUp() override {
        MetaInfoFuncTest::SetUp();
        this->input_info = test_in_info;
        this->model_info = test_model_info;
    }
};

TEST_F(MetaInfoUnitTest, serialize) {
    std::string seriliazation_path(ov::util::path_join({test_artifacts_dir, "test_meta.meta"}));
    this->serialize(seriliazation_path);
    ASSERT_TRUE(ov::util::file_exists(seriliazation_path));

    pugi::xml_document doc;
    doc.load_file(seriliazation_path.c_str());
    {
        auto models_xml = doc.child("meta_info").child("models");
        for (const auto model_xml : models_xml.children()) {
            auto model_name_xml = std::string(model_xml.attribute("name").value());
            ASSERT_NE(model_info.find(model_name_xml), model_info.end());
            ASSERT_EQ(model_info[model_name_xml].this_op_cnt, model_xml.attribute("this_op_count").as_uint());
            ASSERT_EQ(model_info[model_name_xml].total_op_cnt, model_xml.attribute("total_op_count").as_uint());
            auto paths = model_info[model_name_xml].model_paths;
            for (const auto& path_xml : model_xml.child("path")) {
                auto path_xml_value = std::string(path_xml.attribute("path").value());
                ASSERT_NE(std::find(paths.begin(), paths.end(), path_xml_value), paths.end());
            }
        }
    }
    {
        auto graph_priority_xml = doc.child("meta_info").child("graph_priority").attribute("value").as_double();
        ASSERT_EQ(graph_priority_xml, this->get_graph_priority());
    }
    {
        auto input_info_xml = doc.child("meta_info").child("input_info");
        for (const auto& in_info_xml : input_info_xml.children()) {
            auto in_xml = std::string(in_info_xml.attribute("id").value());
            ASSERT_NE(input_info.find(in_xml), input_info.end());
            ASSERT_EQ(input_info[in_xml].is_const, in_info_xml.attribute("convert_to_const").as_bool());
            auto min_xml = std::string(in_info_xml.attribute("min").value()) == "undefined" ? DEFAULT_MIN_VALUE : in_info_xml.attribute("min").as_double();
            ASSERT_EQ(input_info[in_xml].ranges.min, min_xml);
            auto max_xml = std::string(in_info_xml.attribute("max").value()) == "undefined" ? DEFAULT_MAX_VALUE : in_info_xml.attribute("max").as_double();
            ASSERT_EQ(input_info[in_xml].ranges.max, max_xml);
        }
    }
}

TEST_F(MetaInfoUnitTest, update) {
    auto test_meta = MetaInfo(test_model_name, test_in_info);
    std::map<std::string, InputInfo> test_meta_1 = {{ "test_in_0", InputInfo(0, 1, true) }};
    std::string test_model_1 = "test_model_1";
    std::string test_model_path_1 = ov::util::path_join({ "path", "to",  test_model_1 + ".xml"});
    this->update(test_model_path_1, test_meta_1);
    ASSERT_NE(this->model_info.find(test_model_1), this->model_info.end());
    ASSERT_EQ(*this->model_info[test_model_1].model_paths.begin(), test_model_path_1);
    ASSERT_EQ(this->model_info[test_model_1].this_op_cnt, 1);
    this->update(test_model_path_1, test_meta_1);
    ASSERT_EQ(this->model_info[test_model_1].model_paths.size(), 1);
    ASSERT_EQ(this->model_info[test_model_1].this_op_cnt, 2);
    test_model_path_1 = ov::util::path_join({ "path", "to", "test", test_model_1 + ".xml"});
    this->update(test_model_path_1, test_meta_1);
    ASSERT_EQ(this->model_info[test_model_1].model_paths.size(), 2);
    ASSERT_EQ(this->model_info[test_model_1].this_op_cnt, 3);
}

TEST_F(MetaInfoUnitTest, get_model_name_by_path) {
    ASSERT_NO_THROW(this->get_model_name_by_path(test_model_path));
    auto name = this->get_model_name_by_path(test_model_path);
    ASSERT_EQ(name, test_model_name);
}

TEST_F(MetaInfoUnitTest, get_graph_priority) {
    ASSERT_NO_THROW(this->get_graph_priority());
    ASSERT_TRUE(this->get_graph_priority() >= 0 && this->get_graph_priority() <= 1);
    ASSERT_NO_THROW(this->get_abs_graph_priority());
    ASSERT_EQ(this->get_abs_graph_priority(), 5);
}

}  // namespace