// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>

#include <cpu_memory.h>
#include <mutex>
#include <thread>
#include <condition_variable>

using namespace ov::intel_cpu;
using namespace InferenceEngine;

TEST(MemoryTest, SedDataCheck) {
    GTEST_SKIP();
}

TEST(MemoryTest, SedDataWithAutoPadCheck) {
    GTEST_SKIP();
}

TEST(MemoryTest, ConcurrentGetPrimitive) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::memory dnnl_mem1;
    dnnl::memory dnnl_mem2;
    auto desc = std::make_shared<CpuBlockedMemoryDesc>(Precision::FP32, Shape{10, 2});
    Memory cpu_mem1(eng, desc);

    std::atomic<bool> lock{true};

    std::thread worker1([&](){
        while (lock.load()) {}
        dnnl_mem1 = cpu_mem1.getPrimitive();
    });

    std::thread worker2([&](){
        while (lock.load()) {}
        dnnl_mem2 = cpu_mem1.getPrimitive();
    });

    lock.store(false);

    worker1.join();
    worker2.join();
    ASSERT_EQ(dnnl_mem1.get_data_handle(), cpu_mem1.getData());
    ASSERT_EQ(dnnl_mem1, dnnl_mem2);
}

TEST(MemoryTest, ConcurrentResizeGetPrimitive) {
    constexpr size_t number_of_attempts = 10; //just to increase the probability of a collision
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    for (size_t i = 0; i < number_of_attempts; ++i) {
        dnnl::memory dnnl_mem;
        auto desc = std::make_shared<CpuBlockedMemoryDesc>(Precision::FP32, Shape{10, 2});
        Memory cpu_mem1(eng, desc);
        Memory cpu_mem2(eng, desc, cpu_mem1.getMemoryMngr());
        auto desc2 = std::make_shared<CpuBlockedMemoryDesc>(Precision::FP32, Shape{10, 20});

        std::atomic<bool> lock{true};

        std::thread worker1([&](){
            while (lock.load()) {}
            dnnl_mem = cpu_mem1.getPrimitive();
        });

        std::thread worker2([&](){
            while (lock.load()) {}
            cpu_mem2.redefineDesc(desc2);
        });

        lock.store(false);

        worker1.join();
        worker2.join();
        ASSERT_EQ(dnnl_mem.get_data_handle(), cpu_mem2.getData());
    }
}
