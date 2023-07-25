// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/single_layer/ctc_greedy_decoder_seq_len.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
std::string CTCGreedyDecoderSeqLenLayerTest::getTestCaseName(
        const testing::TestParamInfo<ctcGreedyDecoderSeqLenParams>& obj) {
    InferenceEngine::SizeVector inputShape;
    int sequenceLengths;
    InferenceEngine::Precision dataPrecision, indicesPrecision;
    int blankIndex;
    bool mergeRepeated;
    std::string targetDevice;
    std::tie(inputShape,
             sequenceLengths,
             dataPrecision,
             indicesPrecision,
             blankIndex,
             mergeRepeated,
             targetDevice) = obj.param;

    std::ostringstream result;

    result << "IS=" << CommonTestUtils::vec2str(inputShape) << '_';
    result << "seqLen=" << sequenceLengths << '_';
    result << "dataPRC=" << dataPrecision.name() << '_';
    result << "idxPRC=" << indicesPrecision.name() << '_';
    result << "BlankIdx=" << blankIndex << '_';
    result << "mergeRepeated=" << std::boolalpha << mergeRepeated << '_';
    result << "trgDev=" << targetDevice;

    return result.str();
}

void CTCGreedyDecoderSeqLenLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    int sequenceLengths;
    InferenceEngine::Precision dataPrecision, indicesPrecision;
    int blankIndex;
    bool mergeRepeated;
    std::tie(inputShape,
             sequenceLengths,
             dataPrecision,
             indicesPrecision,
             blankIndex,
             mergeRepeated,
             targetDevice) = GetParam();

    auto ngDataPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrecision);
    auto ngIdxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indicesPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngDataPrc, { inputShape });
    auto paramOuts = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));

    const auto sequenceLenNode = [&] {
        const size_t B = inputShape[0];
        const size_t T = inputShape[1];

        // Cap sequence length up to T
        const int seqLen = std::min<int>(T, sequenceLengths);

        std::mt19937 gen{42};
        std::uniform_int_distribution<int> dist(1, seqLen);

        std::vector<int> sequenceLenData(B);
        for (int b = 0; b < B; b++) {
            const int len = dist(gen);
            sequenceLenData[b] = len;
        }

        return ngraph::builder::makeConstant(ngIdxPrc, {B}, sequenceLenData);
    }();

    // Cap blank index up to C - 1
    int C = inputShape.at(2);
    blankIndex = std::min(blankIndex, C - 1);

    auto ctcGreedyDecoderSeqLen = std::dynamic_pointer_cast<ngraph::op::v6::CTCGreedyDecoderSeqLen>(
            ngraph::builder::makeCTCGreedyDecoderSeqLen(paramOuts[0], sequenceLenNode,
                                                        blankIndex, mergeRepeated, ngIdxPrc));

    ngraph::ResultVector results;
    for (int i = 0; i < ctcGreedyDecoderSeqLen->get_output_size(); i++) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoderSeqLen->output(i)));
    }
    function = std::make_shared<ngraph::Function>(results, paramsIn, "CTCGreedyDecoderSeqLen");
}
}  // namespace LayerTestsDefinitions
