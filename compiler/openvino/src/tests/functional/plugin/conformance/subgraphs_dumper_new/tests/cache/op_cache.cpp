// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "gtest/gtest.h"

#include "openvino/op/ops.hpp"
#include "openvino/util/file_util.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"

#include "cache/op_cache.hpp"
#include "utils/node.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ====================== Operation Cache Functional tests ==============================

class OpCacheFuncTest : public ::testing::Test {
protected:
    std::shared_ptr<ov::Model> test_model;
    std::string test_artifacts_dir, test_model_name, test_model_path;

    void SetUp() override {
        test_model_name = "test_model_name";
        test_artifacts_dir = ov::util::path_join({CommonTestUtils::getCurrentWorkingDir(), "test_artifacts"});
        test_model_path = ov::util::path_join({test_artifacts_dir, test_model_name + ".xml"});
        ov::util::create_directory_recursive(test_artifacts_dir);
        {
            auto params = ov::ParameterVector {
                std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::PartialShape{1, 1, 1, 1}),
            };
            auto convert = std::make_shared<ov::op::v0::Convert>(params.front(), ov::element::f16);
            convert->set_friendly_name("convert_0");
            test_model = std::make_shared<ov::Model>(convert, params);
            test_model->set_friendly_name(test_model_name);
        }
    };

    void TearDown() override {
        CommonTestUtils::removeDir(test_artifacts_dir);
        OpCache::reset();
    }
};

TEST_F(OpCacheFuncTest, get_op_cache) {
    std::shared_ptr<ov::tools::subgraph_dumper::OpCache> op_cache = nullptr;
    EXPECT_NO_THROW(op_cache = ov::tools::subgraph_dumper::OpCache::get());
    ASSERT_NE(op_cache, nullptr);
}

TEST_F(OpCacheFuncTest, get_op_cache_twice) {
    std::shared_ptr<ov::tools::subgraph_dumper::OpCache> op_cache_0 = nullptr, op_cache_1 = nullptr;
    op_cache_0 = ov::tools::subgraph_dumper::OpCache::OpCache::get();
    op_cache_1 = ov::tools::subgraph_dumper::OpCache::OpCache::get();
    ASSERT_EQ(op_cache_0, op_cache_1);
}

TEST_F(OpCacheFuncTest, update_cache) {
    auto op_cache = ov::tools::subgraph_dumper::OpCache::get();
    ASSERT_NO_THROW(op_cache->update_cache(test_model, test_model_path, true));
    ASSERT_NO_THROW(op_cache->update_cache(test_model, test_model_path, true));
}

TEST_F(OpCacheFuncTest, serialize_cache) {
    auto op_cache = ov::tools::subgraph_dumper::OpCache::get();
    op_cache->set_serialization_dir(test_artifacts_dir);
    ASSERT_NO_THROW(op_cache->serialize_cache());
}

// ====================== Operation Cache Unit tests ==============================

class OpCacheUnitTest : public OpCacheFuncTest,
                        public virtual OpCache {
protected:
    std::shared_ptr<ov::op::v0::Convert> convert_node;
    MetaInfo test_meta;

    void SetUp() override {
        OpCacheFuncTest::SetUp();
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::PartialShape{1, 1, 1, 1});
        convert_node = std::make_shared<ov::op::v0::Convert>(param, ov::element::f16);
        convert_node->set_friendly_name("convert_0");
        test_meta = MetaInfo(test_model_path, {{"in_0", InputInfo()}});
    }
};

TEST_F(OpCacheUnitTest, update_cache_by_op) {
    this->update_cache(convert_node, test_model_path);
    ASSERT_EQ(m_ops_cache.size(), 1);
}

TEST_F(OpCacheUnitTest, update_cache_by_model) {
    this->update_cache(convert_node, test_model_path, 1);
    ASSERT_EQ(m_ops_cache.size(), 1);
    std::shared_ptr<ov::Model> test_model_1;
    std::string test_model_path_1 = ov::util::path_join({test_artifacts_dir, "model_1", test_model_name + ".xml"});
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::PartialShape{1, 1, 1, 1});
        param->set_friendly_name("in_0");
        auto convert = std::make_shared<ov::op::v0::Convert>(param, ov::element::f16);
        convert->set_friendly_name("convert_0");
        auto erf = std::make_shared<ov::op::v0::Erf>(convert);
        erf->set_friendly_name("erf_0");
        test_model_1 = std::make_shared<ov::Model>(erf, ov::ParameterVector{param});
        test_model_1->set_friendly_name(test_model_name);
    }
    this->update_cache(test_model_1, test_model_path_1, false);
    // check cache
    ASSERT_EQ(m_ops_cache.size(), 2);
    for (const auto& cached_node : this->m_ops_cache) {
        ASSERT_TRUE(std::dynamic_pointer_cast<ov::op::v0::Convert>(cached_node.first) ||
                    std::dynamic_pointer_cast<ov::op::v0::Erf>(cached_node.first));
        auto meta = cached_node.second;
        if (std::dynamic_pointer_cast<ov::op::v0::Convert>(cached_node.first)) {
            // check model_path
            ASSERT_EQ(meta.get_model_info().size(), 1);
            ASSERT_EQ(meta.get_model_info().begin()->first, test_model_name);
            ASSERT_EQ(meta.get_model_info().begin()->second.model_paths.size(), 2);
            ASSERT_EQ(*meta.get_model_info().begin()->second.model_paths.begin(), test_model_path_1);
            ASSERT_EQ(*meta.get_model_info().begin()->second.model_paths.rbegin(), test_model_path);
            // check occurence
            ASSERT_EQ(meta.get_model_info().begin()->second.this_op_cnt, 2);
            ASSERT_EQ(meta.get_model_info().begin()->second.total_op_cnt, 3);
            // check input_info
            ASSERT_EQ(meta.get_input_info().size(), 1);
            ASSERT_EQ(meta.get_input_info().begin()->first, "Convert-1_0");
            ASSERT_EQ(meta.get_input_info().begin()->second.ranges.max, DEFAULT_MAX_VALUE);
            ASSERT_EQ(meta.get_input_info().begin()->second.ranges.min, DEFAULT_MIN_VALUE);
            ASSERT_EQ(meta.get_input_info().begin()->second.is_const, false);
        } else {
            // check model_path
            ASSERT_EQ(meta.get_model_info().size(), 1);
            ASSERT_EQ(meta.get_model_info().begin()->first, test_model_name);
            ASSERT_EQ(meta.get_model_info().begin()->second.model_paths.size(), 1);
            ASSERT_EQ(*meta.get_model_info().begin()->second.model_paths.begin(), test_model_path_1);
            // check occurence
            ASSERT_EQ(meta.get_model_info().begin()->second.this_op_cnt, 1);
            ASSERT_EQ(meta.get_model_info().begin()->second.total_op_cnt, 2);
            // check input_info
            ASSERT_EQ(meta.get_input_info().size(), 1);
            ASSERT_EQ(meta.get_input_info().begin()->first, "Erf-1_0");
            ASSERT_EQ(meta.get_input_info().begin()->second.ranges.max, DEFAULT_MAX_VALUE);
            ASSERT_EQ(meta.get_input_info().begin()->second.ranges.min, DEFAULT_MIN_VALUE);
            ASSERT_EQ(meta.get_input_info().begin()->second.is_const, false);
        }
    }
}

TEST_F(OpCacheUnitTest, serialize_op) {
    this->set_serialization_dir(test_artifacts_dir);
    ASSERT_TRUE(this->serialize_op({convert_node, test_meta}));
    ASSERT_TRUE(ov::util::directory_exists(test_artifacts_dir));
    auto serialized_model_path = ov::util::path_join({test_artifacts_dir,
        "operation", "static", "Convert-1", "f16", "Convert-1_0.xml"});
    ASSERT_TRUE(ov::util::file_exists(serialized_model_path));
    auto core = ov::Core();
    auto serialized_model = core.read_model(serialized_model_path);
    auto res = compare_functions(test_model, serialized_model, true, false, true, true, true, false);
    ASSERT_TRUE(res.first);
}

TEST_F(OpCacheUnitTest, get_rel_serilization_dir) {
    auto ref_path = ov::util::path_join({"operation", "static", "Convert-1", "f16"});
    auto original_path = this->get_rel_serilization_dir(convert_node);
    ASSERT_EQ(ref_path, original_path);
}

TEST_F(OpCacheUnitTest, generate_model_by_node) {
    auto generated_graph = generate_model_by_node(convert_node);
    auto res = compare_functions(test_model, generated_graph, true, false, true, true, true, false);
    ASSERT_TRUE(res.first);
}

}  // namespace
