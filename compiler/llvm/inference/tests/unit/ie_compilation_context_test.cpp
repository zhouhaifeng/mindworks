// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <string>
#include <thread>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "compilation_context.hpp"
#include "cpp/ie_cnn_network.h"
#include "ngraph/function.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

using namespace InferenceEngine;
using namespace ov;
using namespace ngraph;
using namespace ::testing;
using namespace std::chrono;

class FileGuard {
    std::string m_fileName;

public:
    explicit FileGuard(std::string name) : m_fileName(std::move(name)) {}
    ~FileGuard() {
        std::remove(m_fileName.c_str());
    }
};

class NetworkContext_CalcFileInfoTests : public Test {
public:
    std::string m_fileName = "test.blob";

    static void createFile(const std::string& fileName, std::size_t size = 1) {
        std::ofstream str(fileName, std::ios::binary);
        if (!str.good()) {
            GTEST_SKIP();
        }
        for (std::size_t i = 0; i < size; i++)
            str.put('a');
    }

    // Sets up the test fixture.
    void SetUp() override {
        auto testName = CommonTestUtils::generateTestFilePrefix();
        m_fileName = testName + m_fileName;
        createFile(m_fileName);
    }

    // Tears down the test fixture.
    void TearDown() override {
        std::remove(m_fileName.c_str());
    }
};

TEST_F(NetworkContext_CalcFileInfoTests, NoFile) {
    ASSERT_NE(ModelCache::calculate_file_info("notexisting.abc"), ModelCache::calculate_file_info("notexisting2.abc"));

    std::string fileName(100, 'a');
    std::string fileName2(fileName);
    ASSERT_EQ(ModelCache::calculate_file_info(fileName), ModelCache::calculate_file_info(fileName2));
}

TEST_F(NetworkContext_CalcFileInfoTests, ExistingFile) {
    ASSERT_EQ(ModelCache::calculate_file_info(m_fileName), ModelCache::calculate_file_info(m_fileName));
}

TEST_F(NetworkContext_CalcFileInfoTests, ExistingDiffFiles) {
    auto hash1 = ModelCache::calculate_file_info(m_fileName);
    std::string newName = m_fileName + "2";
    std::rename(m_fileName.c_str(), newName.c_str());
    m_fileName = std::move(newName);
    auto hash2 = ModelCache::calculate_file_info(m_fileName);
    ASSERT_NE(hash1, hash2);
}

TEST_F(NetworkContext_CalcFileInfoTests, ExistingFile_sameAbsPath) {
    std::string file1 = m_fileName;
    std::string file2 = std::string(".") + CommonTestUtils::FileSeparator + m_fileName;
    ASSERT_EQ(ModelCache::calculate_file_info(file1), ModelCache::calculate_file_info(file2))
        << "Hash of [" << file1 << "] is not equal to hash of [" << file2 << "]";
}

TEST_F(NetworkContext_CalcFileInfoTests, DateModified) {
    auto info1 = ModelCache::calculate_file_info(m_fileName);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    createFile(m_fileName);
    auto info2 = ModelCache::calculate_file_info(m_fileName);
    ASSERT_NE(info1, info2);
}

TEST_F(NetworkContext_CalcFileInfoTests, SizeModified) {
    createFile(m_fileName, 1);
    auto info1 = ModelCache::calculate_file_info(m_fileName);
    createFile(m_fileName, 2);
    auto info2 = ModelCache::calculate_file_info(m_fileName);
    ASSERT_NE(info1, info2);
}

////////////////////////////////////////////////////

static std::shared_ptr<ngraph::Function> create_simple_function() {
    // This example is taken from docs, shows how to create ngraph::Function
    //
    // Parameter--->Multiply--->Add--->Result
    //    Constant---'          /
    //              Constant---'

    // Create opset6::Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i8, ngraph::Shape{3, 1, 2});
    data->set_friendly_name("Parameter");
    data->get_output_tensor(0).set_names({"parameter"});

    auto mul_constant = ngraph::opset6::Constant::create(ngraph::element::i8, ngraph::Shape{1}, {3});
    mul_constant->set_friendly_name("mul_constant");
    mul_constant->get_output_tensor(0).set_names({"mul_constant"});
    auto mul = std::make_shared<ngraph::opset6::Multiply>(data, mul_constant);
    mul->set_friendly_name("mul");
    mul->get_output_tensor(0).set_names({"mul"});

    auto add_constant = ngraph::opset6::Constant::create(ngraph::element::i8, ngraph::Shape{1}, {2});
    add_constant->set_friendly_name("add_constant");
    add_constant->get_output_tensor(0).set_names({"add_constant"});
    auto add = std::make_shared<ngraph::opset6::Add>(mul, add_constant);
    add->set_friendly_name("add");
    add->get_output_tensor(0).set_names({"add"});

    // Create opset3::Result operation
    auto res = std::make_shared<ngraph::opset6::Result>(add);
    res->set_friendly_name("res");

    // Create nGraph function
    auto func = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
    return func;
}

static void checkCustomRt(const std::function<void(Node::RTMap&)>& emptyCb,
                          const std::function<void(Node::RTMap&, const std::string& name)>& nameCb) {
    auto model1 = create_simple_function();
    auto model2 = create_simple_function();
    auto& op1 = model1->get_ops().front()->get_rt_info();
    auto& op2 = model2->get_ops().front()->get_rt_info();

    emptyCb(op2);
    ASSERT_NE(ModelCache::compute_hash(model1, {}), ModelCache::compute_hash(model2, {}));

    emptyCb(op1);
    ASSERT_EQ(ModelCache::compute_hash(model1, {}), ModelCache::compute_hash(model2, {}));

    nameCb(op1, "test");
    ASSERT_NE(ModelCache::compute_hash(model1, {}), ModelCache::compute_hash(model2, {}));

    nameCb(op2, "test");
    ASSERT_EQ(ModelCache::compute_hash(model1, {}), ModelCache::compute_hash(model2, {}));

    nameCb(op1, "test2");
    ASSERT_NE(ModelCache::compute_hash(model1, {}), ModelCache::compute_hash(model2, {}));
}

TEST(NetworkContext, HashOfSame) {
    auto model1 = create_simple_function();
    auto model2 = create_simple_function();
    ASSERT_EQ(ModelCache::compute_hash(model1, {}), ModelCache::compute_hash(model2, {}));
}

TEST(NetworkContext, HashWithConfig) {
    auto net1 = create_simple_function();
    auto net2 = create_simple_function();
    ASSERT_NE(ModelCache::compute_hash(net1, {{"key", "value"}}), ModelCache::compute_hash(net2, {}));
    ASSERT_EQ(ModelCache::compute_hash(net1, {{"key", "value"}}), ModelCache::compute_hash(net2, {{"key", "value"}}));
}

TEST(NetworkContext, HashWithPrimitivesPriority) {
    auto net1 = create_simple_function();
    auto net2 = create_simple_function();
    auto net3 = create_simple_function();
    auto& op2 = net2->get_ops().front()->get_rt_info();
    op2[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("testPriority");

    auto& op3 = net3->get_ops().front()->get_rt_info();
    op3["PrimitivesPriority"] = "testPriority";

    ASSERT_NE(ModelCache::compute_hash(net1, {}), ModelCache::compute_hash(net2, {}));

    ASSERT_EQ(ModelCache::compute_hash(net2, {}), ModelCache::compute_hash(net3, {}));
}

TEST(NetworkContext, HashWithFusedNames) {
    auto setFusedEmpty = [&](Node::RTMap& rtInfo) {
        rtInfo[ov::FusedNames::get_type_info_static()] = ov::FusedNames();
    };
    auto setFused = [&](Node::RTMap& rtInfo, const std::string& name) {
        rtInfo[ov::FusedNames::get_type_info_static()] = ov::FusedNames(name);
    };
    checkCustomRt(setFusedEmpty, setFused);
}

TEST(NetworkContext, HashWithPrimitivesPriorityType) {
    auto setPrimEmpty = [&](Node::RTMap& rtInfo) {
        rtInfo[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("");
    };
    auto setPrim = [&](Node::RTMap& rtInfo, const std::string& name) {
        rtInfo[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority(name);
    };
    checkCustomRt(setPrimEmpty, setPrim);
}

TEST(NetworkContext, HashWithAffinity) {
    auto net1 = create_simple_function();
    auto net2 = create_simple_function();
    auto net3 = create_simple_function();
    auto& op2 = net2->get_ops().front()->get_rt_info();
    op2["affinity"] = "testAffinity";

    auto& op3 = net3->get_ops().front()->get_rt_info();
    op3["affinity"] = "testAffinity";

    ASSERT_NE(ModelCache::compute_hash(net1, {}), ModelCache::compute_hash(net2, {}));

    ASSERT_EQ(ModelCache::compute_hash(net2, {}), ModelCache::compute_hash(net3, {}));
}

TEST(NetworkContext, HashWithFutureRt_string) {
    auto net1 = create_simple_function();
    auto net2 = create_simple_function();
    auto net3 = create_simple_function();

    auto& op1 = net1->get_ops().front()->get_rt_info();
    op1["someFutureKey"] = "hello";

    auto& op2 = net2->get_ops().front()->get_rt_info();
    op2["someFutureKey"] = "hello";

    auto& op3 = net3->get_ops().front()->get_rt_info();
    op3["someFutureKey"] = "olleh";

    ASSERT_EQ(ModelCache::compute_hash(net1, {}), ModelCache::compute_hash(net2, {}));

    ASSERT_NE(ModelCache::compute_hash(net2, {}), ModelCache::compute_hash(net3, {}));
}

TEST(NetworkContext, HashWithFutureRt_int64) {
    auto net1 = create_simple_function();
    auto net2 = create_simple_function();
    auto net3 = create_simple_function();

    auto& op1 = net1->get_ops().front()->get_rt_info();
    op1["someFutureKey"] = int64_t(42);

    auto& op2 = net2->get_ops().front()->get_rt_info();
    op2["someFutureKey"] = int64_t(42);

    auto& op3 = net3->get_ops().front()->get_rt_info();
    op3["someFutureKey"] = int64_t(43);

    ASSERT_EQ(ModelCache::compute_hash(net1, {}), ModelCache::compute_hash(net2, {}));

    ASSERT_NE(ModelCache::compute_hash(net2, {}), ModelCache::compute_hash(net3, {}));
}

TEST(NetworkContext, HashWithTensorNames) {
    auto fun1 = create_simple_function();
    auto fun2 = create_simple_function();
    auto fun3 = create_simple_function();
    std::unordered_set<std::string> names1, names2;
    std::vector<std::string> testNames;
    testNames.reserve(100);
    for (int i = 0; i < 100; i++) {
        testNames.push_back("test" + std::to_string(i));
    }
    std::for_each(testNames.begin(), testNames.end(), [&names1](const std::string& name) {
        names1.insert(name);
    });
    std::for_each(testNames.rbegin(), testNames.rend(), [&names2](const std::string& name) {
        names2.insert(name);
    });

    fun1->input().set_names(names1);
    fun2->input().set_names(names2);

    ASSERT_EQ(ModelCache::compute_hash(fun1, {}), ModelCache::compute_hash(fun2, {}));

    ASSERT_NE(ModelCache::compute_hash(fun2, {}), ModelCache::compute_hash(fun3, {}));
}

TEST(NetworkContext, HashWithDifferentResults) {
    auto net1 = create_simple_function();
    auto net2 = create_simple_function();
    net2->remove_result(net2->get_results().front());
    auto net3 = create_simple_function();
    net3->remove_result(net3->get_results().front());
    ASSERT_NE(ModelCache::compute_hash(net1, {}), ModelCache::compute_hash(net2, {}));
    ASSERT_EQ(ModelCache::compute_hash(net2, {}), ModelCache::compute_hash(net3, {}));
}

// Verify all internal hash calculations are thread-safe (like ngraph::function serialization)
TEST(NetworkContext, HashOfSameMultiThreading) {
    auto net1 = create_simple_function();
    auto net2 = create_simple_function();
    std::atomic_bool fail{false};
    const auto TEST_DURATION_MS = 1000;
    auto start = high_resolution_clock::now();
    int t1Count = 0, t2Count = 0;
    auto threadFun = [&](int& count) {
        do {
            count++;
            auto hash1 = ModelCache::compute_hash(net1, {});
            auto hash2 = ModelCache::compute_hash(net2, {});
            if (hash1 != hash2) {
                fail = true;
                break;
            }
        } while (!fail && duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    };
    std::thread t1(threadFun, std::ref(t1Count));
    std::thread t2(threadFun, std::ref(t2Count));
    t1.join();
    t2.join();
    std::cout << "Hash threading test finished. Total runs = " << t1Count + t2Count << std::endl;
    ASSERT_FALSE(fail);
}

////////////////////////////////////////////

TEST(NetworkContext_ModelName, HashOfSame) {
    ASSERT_EQ(ModelCache::compute_hash("model1", {}), ModelCache::compute_hash("model1", {}));

    ASSERT_NE(ModelCache::compute_hash("model1", {}), ModelCache::compute_hash("model2", {}));

    ASSERT_NE(ModelCache::compute_hash("model1", {{"key", "value"}}), ModelCache::compute_hash("model1", {}));

    ASSERT_EQ(ModelCache::compute_hash("model1", {{"key", "value"}}),
              ModelCache::compute_hash("model1", {{"key", "value"}}));
}

TEST(NetworkContext_ModelName, HashOfExistingFile) {
    auto file1 = CommonTestUtils::generateTestFilePrefix() + ".xml";
    auto file2 = std::string(".") + CommonTestUtils::FileSeparator + file1;

    FileGuard guard(file1);
    {
        std::ofstream os(file1);
        os << "test";
    }
    ASSERT_EQ(ModelCache::compute_hash(file1, {}), ModelCache::compute_hash(file1, {}));

    ASSERT_EQ(ModelCache::compute_hash(file1, {}), ModelCache::compute_hash(file2, {}));

    ASSERT_NE(ModelCache::compute_hash(file1, {{"key", "value"}}), ModelCache::compute_hash(file2, {}));

    ASSERT_EQ(ModelCache::compute_hash(file1, {{"key", "value"}}), ModelCache::compute_hash(file2, {{"key", "value"}}));
}
