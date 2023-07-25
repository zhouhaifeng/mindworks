// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/deformable_psroi_pooling.hpp"


namespace LayerTestsDefinitions {

    std::string DeformablePSROIPoolingLayerTest::getTestCaseName(const testing::TestParamInfo<deformablePSROILayerTestParams>& obj) {
        std::vector<size_t> dataShape;
        std::vector<size_t> roisShape;
        std::vector<size_t> offsetsShape;
        int64_t outputDim;
        int64_t groupSize;
        float spatialScale;
        std::vector<int64_t> spatialBinsXY;
        float trans_std;
        int64_t part_size;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        deformablePSROISpecificParams opParams;

        std::tie(opParams, netPrecision, targetDevice) = obj.param;
        std::tie(dataShape, roisShape, offsetsShape, outputDim, groupSize, spatialScale, spatialBinsXY,
        trans_std, part_size) = opParams;

        std::ostringstream result;

        result << "data_shape=" << CommonTestUtils::vec2str(dataShape) << "_";
        result << "rois_shape=" << CommonTestUtils::vec2str(roisShape) << "_";
        result << "offsets_shape=" << CommonTestUtils::vec2str(offsetsShape) << "_";
        result << "out_dim=" << outputDim << "_";
        result << "group_size=" << groupSize << "_";
        result << "scale=" << spatialScale << "_";
        result << "bins_x=" << spatialBinsXY[0] << "_";
        result << "bins_y=" << spatialBinsXY[1] << "_";
        result << "trans_std=" << trans_std << "_";
        result << "part_size=" << part_size << "_";
        result << "prec=" << netPrecision.name() << "_";
        result << "dev=" << targetDevice;
        return result.str();
    }

    void DeformablePSROIPoolingLayerTest::GenerateInputs() {
        auto data_input_shape = cnnNetwork.getInputShapes().begin()->second;
        const auto batch_distrib = data_input_shape[0] - 1;
        const auto height = data_input_shape[2] / spatialScale_;
        const auto width  = data_input_shape[3] / spatialScale_;

        size_t it = 0;
        for (const auto &input : cnnNetwork.getInputsInfo()) {
            const auto &info = input.second;
            InferenceEngine::Blob::Ptr blob;

            if (it == 0) {
                blob = GenerateInput(*info);
            } else if (it == 1) {
                blob = make_blob_with_precision(info->getTensorDesc());
                blob->allocate();
                CommonTestUtils::fill_data_roi<InferenceEngine::Precision::FP32>(blob, batch_distrib,
                                               height, width, 1.0f, true);
            } else {
                blob = make_blob_with_precision(info->getTensorDesc());
                blob->allocate();
                std::vector<float> offset_data = CommonTestUtils::generate_float_numbers(blob->size(), -0.9, 0.9);
                CommonTestUtils::fill_data_float_array<InferenceEngine::Precision::FP32>(blob, &offset_data[0], blob->size());
            }
            inputs.push_back(blob);
            it++;
        }
    }

    void DeformablePSROIPoolingLayerTest::SetUp() {
        std::vector<size_t> dataShape;
        std::vector<size_t> roisShape;
        std::vector<size_t> offsetsShape;
        int64_t outputDim;
        int64_t groupSize;
        std::string mode = "bilinear_deformable";
        std::vector<int64_t> spatialBinsXY;
        float trans_std;
        int64_t part_size;
        InferenceEngine::Precision netPrecision;
        deformablePSROISpecificParams opParams;

        std::tie(opParams, netPrecision, targetDevice) = this->GetParam();
        std::tie(dataShape, roisShape, offsetsShape, outputDim, groupSize, spatialScale_, spatialBinsXY,
        trans_std, part_size) = opParams;


        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ngraph::ParameterVector params;
        ngraph::OutputVector inputs;
        std::shared_ptr<ngraph::Node> defomablePSROIPooling;

        if (offsetsShape.empty()) { // Test without optional third input (offsets)
            params = ngraph::builder::makeParams(ngPrc, {dataShape, roisShape});
            inputs = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
            defomablePSROIPooling = std::make_shared<ngraph::op::v1::DeformablePSROIPooling>(inputs[0],
                                                                                                inputs[1],
                                                                                                outputDim,
                                                                                                spatialScale_,
                                                                                                groupSize,
                                                                                                mode,
                                                                                                spatialBinsXY[0],
                                                                                                spatialBinsXY[1],
                                                                                                trans_std,
                                                                                                part_size);
        } else {
            params = ngraph::builder::makeParams(ngPrc, {dataShape, roisShape, offsetsShape});
            inputs = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
            defomablePSROIPooling = std::make_shared<ngraph::op::v1::DeformablePSROIPooling>(inputs[0],
                                                                                                inputs[1],
                                                                                                inputs[2],
                                                                                                outputDim,
                                                                                                spatialScale_,
                                                                                                groupSize,
                                                                                                mode,
                                                                                                spatialBinsXY[0],
                                                                                                spatialBinsXY[1],
                                                                                                trans_std,
                                                                                                part_size);
        }

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(defomablePSROIPooling)};
        function = std::make_shared<ngraph::Function>(results, params, "deformable_psroi_pooling");
    }
}  // namespace LayerTestsDefinitions
