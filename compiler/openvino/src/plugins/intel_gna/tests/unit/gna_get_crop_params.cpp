// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>
// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include <legacy/ie_layers.h>

#include <layers/gna_crop_layer.hpp>

namespace {

using namespace ov::intel_gna;

typedef std::tuple<std::vector<size_t>,  // Input shape
                   std::vector<int>,     // Output shape
                   std::vector<int>,     // Axes
                   std::vector<int>,     // Offset
                   size_t,               // Offset in flatten data
                   size_t                // Output size in flatten data
                   >
    CropParams;

const std::vector<CropParams> crop_params_vector = {{{8, 16}, {1, 16}, {0, 1}, {1, 0}, 16, 16},
                                                    {{5, 24}, {1, 24}, {0, 1}, {2, 0}, 48, 24},
                                                    {{8, 16}, {2, 16}, {0, 1}, {1, 0}, 16, 32},
                                                    {{1, 16}, {1, 8}, {0, 1}, {0, 5}, 5, 8},
                                                    {{1, 8, 16}, {1, 1, 16}, {0, 1, 2}, {0, 1, 0}, 16, 16},
                                                    {{1, 1, 8, 16}, {1, 1, 1, 16}, {0, 1, 2, 3}, {0, 0, 1, 0}, 16, 16}};

TEST(GetCropParamsTest, testGetCropParams) {
    InferenceEngine::LayerParams attrs = {"Crop", "Crop", InferenceEngine::Precision::FP32};
    for (const auto& crop_params : crop_params_vector) {
        std::vector<size_t> in_shape;
        std::vector<int> orig_out_shape, orig_axes, orig_offset;
        size_t result_offset, result_out_size;
        std::tie(in_shape, orig_out_shape, orig_axes, orig_offset, result_offset, result_out_size) = crop_params;

        auto crop_layer = std::make_shared<InferenceEngine::CropLayer>(attrs);
        auto layout = in_shape.size() == 2 ? InferenceEngine::NC
                                           : (in_shape.size() == 3 ? InferenceEngine::CHW : InferenceEngine::NCHW);
        auto data = std::make_shared<InferenceEngine::Data>(
            "Crop_input",
            InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, in_shape, layout));
        crop_layer->insData.push_back(data);
        crop_layer->dim = orig_out_shape;
        crop_layer->axis = orig_axes;
        crop_layer->offset = orig_offset;
        const auto results = GetCropParams(crop_layer.get());
        ASSERT_EQ(results.start_offset, result_offset);
        ASSERT_EQ(results.crop_size, result_out_size);
    }
}

}  // namespace