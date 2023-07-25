// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "gtest/gtest.h"

#include "openvino/op/ops.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/op/util/op_types.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"

#include "cache/graph_cache.hpp"
#include "utils/node.hpp"
#include "test_models/model_0.hpp"
#include "test_models/model_1.hpp"
#include "test_models/model_2.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ====================== Graph Cache Functional tests ==============================

class GraphCacheFuncTest : public ::testing::Test {
protected:
    std::shared_ptr<ov::Model> test_model;
    std::string test_artifacts_dir, test_model_name, test_model_path;

    void SetUp() override {
        test_model_name = "test_model_name";
        test_artifacts_dir = ov::util::path_join({CommonTestUtils::getCurrentWorkingDir(), "test_artifacts"});
        test_model_path = ov::util::path_join({test_artifacts_dir, test_model_name + ".xml"});
        ov::util::create_directory_recursive(test_artifacts_dir);
        {
            Model_0 test;
            test_model = test.get();
            test_model->set_friendly_name(test_model_name);
        }
    };

    void TearDown() override {
        CommonTestUtils::removeDir(test_artifacts_dir);
        GraphCache::reset();
    }
};

TEST_F(GraphCacheFuncTest, get_graph_cache) {
    std::shared_ptr<ov::tools::subgraph_dumper::GraphCache> graph_cache = nullptr;
    EXPECT_NO_THROW(graph_cache = ov::tools::subgraph_dumper::GraphCache::get());
    ASSERT_NE(graph_cache, nullptr);
}

TEST_F(GraphCacheFuncTest, get_graph_cache_twice) {
    std::shared_ptr<ov::tools::subgraph_dumper::GraphCache> graph_cache_0 = nullptr, graph_cache_1 = nullptr;
    graph_cache_0 = ov::tools::subgraph_dumper::GraphCache::get();
    graph_cache_1 = ov::tools::subgraph_dumper::GraphCache::get();
    ASSERT_EQ(graph_cache_0, graph_cache_1);
}

TEST_F(GraphCacheFuncTest, update_cache) {
    auto graph_cache = ov::tools::subgraph_dumper::GraphCache::get();
    ASSERT_NO_THROW(graph_cache->update_cache(test_model, test_model_path, true));
    ASSERT_NO_THROW(graph_cache->update_cache(test_model, test_model_path, true));
}

TEST_F(GraphCacheFuncTest, serialize_cache) {
    auto graph_cache = ov::tools::subgraph_dumper::GraphCache::get();
    graph_cache->set_serialization_dir(test_artifacts_dir);
    ASSERT_NO_THROW(graph_cache->serialize_cache());
}

// ====================== Graph Cache Unit tests ==============================

class GraphCacheUnitTest : public GraphCacheFuncTest,
                           public virtual GraphCache {
protected:
    std::shared_ptr<ov::op::v0::Convert> convert_node;
    MetaInfo test_meta;

    void SetUp() override {
        GraphCacheFuncTest::SetUp();
    }
};

TEST_F(GraphCacheUnitTest, update_cache_by_graph) {
    // const std::shared_ptr<ov::Model>& model, const std::string& model_path,
                    //   const std::map<std::string, InputInfo>& input_info, size_t model_op_cnt
    Model_2 test;
    auto model_to_cache = test.get();
    std::map<std::string, InputInfo> in_info;
    for (const auto& op : model_to_cache->get_ordered_ops()) {
        if (ov::op::util::is_parameter(op)) {
            in_info.insert({ op->get_friendly_name(), InputInfo()});
        }
    }
    this->update_cache(model_to_cache, test_model_path, in_info, model_to_cache->get_ordered_ops().size());
    ASSERT_EQ(m_graph_cache.size(), 1);
}
}  // namespace
