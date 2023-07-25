// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory/gna_memory.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace memory;

class GNAMemoryTest : public ::testing::Test {
protected:
    GNAMemory<GNAFloatAllocator> mem{GNAFloatAllocator{}};

    void SetUp() override {}
};

TEST_F(GNAMemoryTest, canStoreActualBlob) {
    float input[] = {1, 2, 3};
    float* pFuture = nullptr;
    size_t len = sizeof(input);

    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);
    mem.commit();

    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFuture, input);
    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
}

TEST_F(GNAMemoryTest, canStore2Blobs) {
    float input[] = {1, 2, 3, 4};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;

    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, 3 * 4);
    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture2, input + 1, 3 * 4);
    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, input);
    ASSERT_EQ(pFuture + 3, pFuture2);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
    ASSERT_EQ(pFuture[3], 2);
    ASSERT_EQ(pFuture[4], 3);
    ASSERT_EQ(pFuture[5], 4);
}

TEST_F(GNAMemoryTest, canStoreBlobsALIGNED) {
    GNAMemory<memory::GNAFloatAllocator> dataAlignedMem(16);
    float input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float* pFuture = nullptr;
    auto queue = dataAlignedMem.getQueue(REGION_SCRATCH);
    queue->push_ptr(nullptr, &pFuture, input, 3 * 4);
    dataAlignedMem.commit();

    ASSERT_EQ(16, queue->getSize());

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture, nullptr);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
    // least probability for next element to be equal if not copied
    ASSERT_NE(pFuture[3], 4);
}

TEST_F(GNAMemoryTest, canStore2BlobsALIGNED) {
    GNAMemory<memory::GNAFloatAllocator> dataAlignedMem(8);
    float input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;
    auto queue = dataAlignedMem.getQueue(REGION_SCRATCH);
    queue->push_ptr(nullptr, &pFuture, input, 3 * 4);
    queue->push_ptr(nullptr, &pFuture2, input, 1 * 4);
    dataAlignedMem.commit();

    ASSERT_EQ(24, queue->getSize());

    ASSERT_NE(pFuture, nullptr);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
    // least probability for next element to be equal if not copied
    ASSERT_EQ(pFuture[4], 1);
}

TEST_F(GNAMemoryTest, canReserveData) {
    float* pFuture = nullptr;
    mem.getQueue(REGION_SCRATCH)->reserve_ptr(nullptr, &pFuture, 3 * 4);
    mem.commit();

    ASSERT_NE(pFuture, nullptr);
}

TEST_F(GNAMemoryTest, canReserveDataByVoid) {
    mem.getQueue(REGION_SCRATCH)->reserve_ptr(nullptr, nullptr, 3 * 4);
    ASSERT_NO_THROW(mem.commit());
}

TEST_F(GNAMemoryTest, canReserveAndPushData) {
    float input[] = {1, 2, 3};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;
    size_t len = sizeof(input);

    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);
    mem.getQueue(REGION_SCRATCH)->reserve_ptr(nullptr, &pFuture2, 3 * 4);
    mem.commit();

    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFuture2, nullptr);
    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, pFuture);

    pFuture2[0] = -1;
    pFuture2[1] = -1;
    pFuture2[2] = -1;

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
}

TEST_F(GNAMemoryTest, canBindAndResolve) {
    float input[] = {1, 2, 3};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;
    float* pFuture3 = nullptr;
    size_t len = sizeof(input);

    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture3, &pFuture);
    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture2, &pFuture);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, nullptr);
    ASSERT_EQ(pFuture2, pFuture);
    ASSERT_EQ(pFuture3, pFuture);

    ASSERT_EQ(pFuture2[0], 1);
    ASSERT_EQ(pFuture2[1], 2);
    ASSERT_EQ(pFuture2[2], 3);
}

TEST_F(GNAMemoryTest, canBindTransitevlyAndResolve) {
    float input[] = {1, 2, 3};
    float* pFuture = nullptr;
    float* pFuture3 = nullptr;
    float* pFuture4 = nullptr;
    size_t len = sizeof(input);

    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture4, &pFuture3);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture3, &pFuture);
    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_EQ(pFuture3, pFuture);
    ASSERT_EQ(pFuture4, pFuture);

    ASSERT_NE(pFuture4, nullptr);

    ASSERT_EQ(pFuture4[0], 1);
    ASSERT_EQ(pFuture4[1], 2);
    ASSERT_EQ(pFuture4[2], 3);
}

TEST_F(GNAMemoryTest, canBindTransitevlyWithOffsetsAndResolve) {
    float input[] = {1, 2, 3};
    float* pFuture = nullptr;
    float* pFuture3 = nullptr;
    float* pFuture4 = nullptr;
    size_t len = sizeof(input);

    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture4, &pFuture3, 4);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture3, &pFuture, 4);
    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_EQ(pFuture3, pFuture + 1);
    ASSERT_EQ(pFuture4, pFuture + 2);

    ASSERT_NE(pFuture, nullptr);

    ASSERT_EQ(pFuture[0], 1);
    ASSERT_EQ(pFuture[1], 2);
    ASSERT_EQ(pFuture[2], 3);
}

TEST_F(GNAMemoryTest, canBindWithOffsetAndResolve) {
    float input[] = {1, 2, 3};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;
    float* pFuture3 = nullptr;
    size_t len = sizeof(input);

    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture3, &pFuture, 4);
    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture2, &pFuture);

    mem.commit();

    ASSERT_NE(pFuture, input);
    ASSERT_NE(pFuture2, nullptr);
    ASSERT_EQ(pFuture2, pFuture);
    ASSERT_NE(pFuture3, nullptr);
    ASSERT_EQ(pFuture3, pFuture + 1);

    ASSERT_EQ(pFuture2[0], 1);
    ASSERT_EQ(pFuture2[1], 2);
    ASSERT_EQ(pFuture2[2], 3);
    ASSERT_EQ(pFuture3[0], 2);
}

TEST_F(GNAMemoryTest, canPushLocal) {
    float* pFuture = reinterpret_cast<float*>(&pFuture);

    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        mem.getQueue(REGION_SCRATCH)->push_local_ptr(nullptr, pFuture, &*input.begin(), 4 * 4);
    }

    // poison stack
    mem.commit();

    ASSERT_FLOAT_EQ(pFuture[0], 1);
    ASSERT_FLOAT_EQ(pFuture[1], 2);
    ASSERT_FLOAT_EQ(pFuture[2], 3);
    ASSERT_FLOAT_EQ(pFuture[3], 4);
}

TEST_F(GNAMemoryTest, canPushValue) {
    float* pFuture = reinterpret_cast<float*>(&pFuture);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);

    {
        mem.getQueue(REGION_SCRATCH)->push_value(nullptr, pFuture, 3.f, 2);
        mem.getQueue(REGION_SCRATCH)->push_value(nullptr, pFuture2, 13.f, 2);
    }

    mem.commit();

    ASSERT_FLOAT_EQ(pFuture[0], 3);
    ASSERT_FLOAT_EQ(pFuture[1], 3);
    ASSERT_FLOAT_EQ(pFuture[2], 13);
    ASSERT_FLOAT_EQ(pFuture[3], 13);
}

TEST_F(GNAMemoryTest, canPushReadOnlyValue) {
    float* pFuture = reinterpret_cast<float*>(&pFuture);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);

    {
        mem.getQueue(REGION_SCRATCH)->push_value(nullptr, pFuture, 3.f, 2);
        mem.getQueue(REGION_RO)->push_value(nullptr, pFuture2, 13.f, 2);
    }

    mem.commit();

    ASSERT_FLOAT_EQ(pFuture[0], 3);
    ASSERT_FLOAT_EQ(pFuture[1], 3);
    ASSERT_FLOAT_EQ(pFuture2[0], 13);
    ASSERT_FLOAT_EQ(pFuture2[1], 13);
}

TEST_F(GNAMemoryTest, canCalculateReadWriteSectionSizeEmptyReqs) {
    mem.getQueue(REGION_SCRATCH)->push_value(nullptr, nullptr, 3.f, 2);
    mem.getQueue(REGION_RO)->push_value(nullptr, nullptr, 13.f, 2);
    mem.commit();

    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_SCRATCH), 0);
    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_RO), 0);
}

TEST_F(GNAMemoryTest, canCalculateReadWriteSectionSizeWithEmptyReqs) {
    // empty request before
    mem.getQueue(REGION_SCRATCH)->push_value(nullptr, nullptr, 3.f, 2);
    // not empty requests
    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    mem.getQueue(REGION_SCRATCH)->push_value(nullptr, pFuture1, 3.f, 2);
    mem.getQueue(REGION_RO)->push_value(nullptr, pFuture2, 13.f, 2);
    // empty request after
    mem.getQueue(REGION_SCRATCH)->push_value(nullptr, nullptr, 3.f, 2);
    mem.getQueue(REGION_RO)->push_value(nullptr, nullptr, 13.f, 2);
    mem.commit();

    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_RO), 2 * sizeof(float));
    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_SCRATCH), 2 * sizeof(float));
}

TEST_F(GNAMemoryTest, canCalculateReadWriteSectionSize) {
    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    mem.getQueue(REGION_SCRATCH)->push_value(nullptr, pFuture1, 3.f, 2);
    mem.getQueue(REGION_RO)->push_value(nullptr, pFuture2, 13.f, 2);
    mem.commit();

    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_RO), 2 * sizeof(float));
    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_SCRATCH), 2 * sizeof(float));
}

TEST_F(GNAMemoryTest, canCalculateReadWriteSectionSizeWithAlignment) {
    GNAMemory<memory::GNAFloatAllocator> memAligned(1, 64);
    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);

    memAligned.getQueue(REGION_SCRATCH)->push_value(nullptr, pFuture1, 3.f, 2);
    memAligned.getQueue(REGION_RO)->push_value(nullptr, pFuture2, 13.f, 2);
    memAligned.commit();

    ASSERT_EQ(memAligned.getRegionBytes(rRegion::REGION_RO), 64);
    ASSERT_EQ(memAligned.getRegionBytes(rRegion::REGION_SCRATCH), 64);
}

TEST_F(GNAMemoryTest, canSetUpReadWriteSectionPtr) {
    float* pFuture1 = reinterpret_cast<float*>(&pFuture1);
    float* pFuture2 = reinterpret_cast<float*>(&pFuture2);
    float* pFuture3 = reinterpret_cast<float*>(&pFuture3);

    mem.getQueue(REGION_RO)->push_value(nullptr, pFuture1, 3.f, 2);
    mem.getQueue(REGION_SCRATCH)->push_value(nullptr, pFuture2, 13.f, 3);
    mem.getQueue(REGION_RO)->push_value(nullptr, pFuture3, 32.f, 4);
    mem.commit();

    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_RO), (2 + 4) * sizeof(float));
    ASSERT_EQ(mem.getRegionBytes(rRegion::REGION_SCRATCH), 3 * sizeof(float));

    ASSERT_NE(&pFuture2[0], &pFuture1[0]);
    ASSERT_LT(&pFuture1[0], &pFuture3[0]);

    ASSERT_FLOAT_EQ(pFuture1[0], 3.f);
    ASSERT_FLOAT_EQ(pFuture1[1], 3.f);

    ASSERT_FLOAT_EQ(pFuture2[0], 13.f);
    ASSERT_FLOAT_EQ(pFuture2[1], 13.f);
    ASSERT_FLOAT_EQ(pFuture2[2], 13.f);

    ASSERT_FLOAT_EQ(pFuture3[0], 32.f);
    ASSERT_FLOAT_EQ(pFuture3[1], 32.f);
    ASSERT_FLOAT_EQ(pFuture3[2], 32.f);
    ASSERT_FLOAT_EQ(pFuture3[3], 32.f);
}

TEST_F(GNAMemoryTest, canUpdateSizeOfPushRequestWithBindRequest) {
    float input[] = {1, 2, 3};
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;
    float* pFuture3 = nullptr;

    size_t len = sizeof(input);

    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture2, &pFuture, len, len);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture3, &pFuture2, 2 * len, len);

    mem.commit();

    ASSERT_EQ(mem.getRegionBytes(REGION_SCRATCH), 4 * len);
    ASSERT_NE(pFuture, nullptr);
    ASSERT_EQ(pFuture2, pFuture + 3);
    ASSERT_EQ(pFuture3, pFuture + 9);

    ASSERT_FLOAT_EQ(pFuture[0], 1);
    ASSERT_FLOAT_EQ(pFuture[1], 2);
    ASSERT_FLOAT_EQ(pFuture[2], 3);
    ASSERT_FLOAT_EQ(pFuture[3], 0);
    ASSERT_FLOAT_EQ(pFuture[4], 0);
    ASSERT_FLOAT_EQ(pFuture[5], 0);
    ASSERT_FLOAT_EQ(pFuture[6], 0);
    ASSERT_FLOAT_EQ(pFuture[7], 0);
    ASSERT_FLOAT_EQ(pFuture[8], 0);
}

TEST_F(GNAMemoryTest, canUpdateSizeOfPushRequestWithBindRequestWhenPush) {
    float input[] = {1, 2, 3};
    float input2[] = {6, 7, 8};

    float* pFutureInput2 = nullptr;
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;

    size_t len = sizeof(input);

    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFuture, input, len);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture2, &pFuture, len, len);
    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFutureInput2, input2, len);

    mem.commit();

    ASSERT_EQ(mem.getRegionBytes(REGION_SCRATCH), 3 * len);
    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFutureInput2, nullptr);
    ASSERT_EQ(pFuture2, pFuture + 3);

    ASSERT_FLOAT_EQ(pFuture[0], 1);
    ASSERT_FLOAT_EQ(pFuture[1], 2);
    ASSERT_FLOAT_EQ(pFuture[2], 3);
    ASSERT_FLOAT_EQ(pFuture[3], 0);
    ASSERT_FLOAT_EQ(pFuture[4], 0);

    ASSERT_FLOAT_EQ(pFutureInput2[0], 6);
    ASSERT_FLOAT_EQ(pFutureInput2[1], 7);
    ASSERT_FLOAT_EQ(pFutureInput2[2], 8);
}

TEST_F(GNAMemoryTest, canUpdateSizeOfPushRequestWithBindRequestWhenAlloc) {
    float input[] = {1, 2, 3};

    float* pFutureInput = nullptr;
    float* pFuture = nullptr;
    float* pFuture2 = nullptr;

    size_t len = sizeof(input);

    mem.getQueue(REGION_SCRATCH)->reserve_ptr(nullptr, &pFuture, len);
    mem.getQueue(REGION_AUTO)->bind_ptr(nullptr, &pFuture2, &pFuture, len, len);
    mem.getQueue(REGION_SCRATCH)->push_ptr(nullptr, &pFutureInput, input, len);

    mem.commit();

    ASSERT_EQ(mem.getRegionBytes(REGION_SCRATCH), 3 * len);
    ASSERT_NE(pFuture, nullptr);
    ASSERT_NE(pFutureInput, nullptr);
    ASSERT_EQ(pFuture2, pFuture + 3);

    ASSERT_FLOAT_EQ(pFuture[0], 0);
    ASSERT_FLOAT_EQ(pFuture[1], 0);
    ASSERT_FLOAT_EQ(pFuture[2], 0);
    ASSERT_FLOAT_EQ(pFuture[3], 0);
    ASSERT_FLOAT_EQ(pFuture[4], 0);

    ASSERT_FLOAT_EQ(pFutureInput[0], 1);
    ASSERT_FLOAT_EQ(pFutureInput[1], 2);
    ASSERT_FLOAT_EQ(pFutureInput[2], 3);
}
