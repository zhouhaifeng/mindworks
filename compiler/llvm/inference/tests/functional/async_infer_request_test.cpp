// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

IE_SUPPRESS_DEPRECATED_START

TEST(InferRequestCPPTests, throwsOnUninitializedSetBlob) {
    InferRequest req;
    ASSERT_THROW(req.SetBlob({}, {}), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetBlob) {
    InferRequest req;
    ASSERT_THROW(req.GetBlob({}), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetPreProcess) {
    InferRequest req;
    ASSERT_THROW(req.GetPreProcess({}), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedInfer) {
    InferRequest req;
    ASSERT_THROW(req.Infer(), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetPerformanceCounts) {
    InferRequest req;
    ASSERT_THROW(req.GetPerformanceCounts(), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetInput) {
    InferRequest req;
    ASSERT_THROW(req.SetInput({{}}), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetOutput) {
    InferRequest req;
    ASSERT_THROW(req.SetOutput({{}}), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedStartAsync) {
    InferRequest req;
    ASSERT_THROW(req.StartAsync(), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedWait) {
    InferRequest req;
    ASSERT_THROW(req.Wait({}), InferenceEngine::NotAllocated);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetCompletionCallback) {
    InferRequest req;
    std::function<void(InferRequest, StatusCode)> f;
    ASSERT_THROW(req.SetCompletionCallback(f), InferenceEngine::NotAllocated);
}

IE_SUPPRESS_DEPRECATED_START

TEST(InferRequestCPPTests, throwsOnUninitializedCast) {
    InferRequest req;
    ASSERT_THROW((void)static_cast<IInferRequest::Ptr>(req), InferenceEngine::NotAllocated);
}

IE_SUPPRESS_DEPRECATED_END

TEST(InferRequestCPPTests, throwsOnUninitializedQueryState) {
    InferRequest req;
    ASSERT_THROW(req.QueryState(), InferenceEngine::NotAllocated);
}
