// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class EmbeddingBagSum {
public:
    EmbeddingBagSum(
            const std::shared_ptr<ngraph::Node>&,
            size_t requiredInputsNum,
            size_t indicesIdx,
            size_t perSampleWeightsIdx,
            size_t defaultIndexIdx);

    void execute(const uint8_t* srcData, const uint8_t* weightsData, const InferenceEngine::Precision &srcPrc,
                 const InferenceEngine::SizeVector& inDims, const MemoryPtr& outMemory);

    ~EmbeddingBagSum() = default;

protected:
    virtual void initFromInputs() = 0;
    virtual void getIndices(
            size_t embIndex,
            const int*& indicesRef,
            size_t& size,
            int& weightsIdx,
            bool& withWeights) = 0;

    void prepareParams(const VectorDims& indexStaticShape);

    template<typename T>
    void processData(const T* srcData, const T* weightsData,
                     const InferenceEngine::SizeVector& inDataDims, const MemoryPtr& outMemory);

    const size_t EMB_TABLE_IDX = 0lu;
    const size_t INDICES_IDX;
    const size_t PER_SAMPLE_WEIGHTS_IDX;
    const size_t DEFAULT_INDEX_IDX;

    bool _withWeights = false;
    size_t _embDepth = 0;
    std::string _layerName;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
