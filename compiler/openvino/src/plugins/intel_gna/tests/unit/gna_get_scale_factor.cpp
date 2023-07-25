// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <vector>
// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "frontend/scale_factor_calc.hpp"
#include "gna_graph_tools.hpp"

namespace {

class GnaGetScaleFactorTest : public ::testing::Test {
protected:
    void GetScaleFactorAndCheck(float src_scale, float dst_scale, float weights_scale, float bias_scale) const {
        InferenceEngine::LayerParams params("fc", "FullyConnected", InferenceEngine::Precision::FP32);
        InferenceEngine::CNNLayerPtr layer = std::make_shared<InferenceEngine::CNNLayer>(params);
        layer = InferenceEngine::injectData<ov::intel_gna::frontend::QuantizedLayerParams>(*layer);
        auto quant = InferenceEngine::getInjectedData<ov::intel_gna::frontend::QuantizedLayerParams>(*layer);
        quant->_src_quant.SetScale(src_scale);
        quant->_dst_quant.SetScale(dst_scale);
        quant->_weights_quant.SetScale(weights_scale);
        quant->_bias_quant.SetScale(bias_scale);
        ASSERT_EQ(ov::intel_gna::frontend::GetScaleFactor(layer, ov::intel_gna::frontend::QuantizedDataType::input),
                  src_scale);
        ASSERT_EQ(ov::intel_gna::frontend::GetScaleFactor(layer, ov::intel_gna::frontend::QuantizedDataType::output),
                  dst_scale);
        ASSERT_EQ(ov::intel_gna::frontend::GetScaleFactor(layer, ov::intel_gna::frontend::QuantizedDataType::weights),
                  weights_scale);
        ASSERT_EQ(ov::intel_gna::frontend::GetScaleFactor(layer, ov::intel_gna::frontend::QuantizedDataType::bias),
                  bias_scale);
    }
};

TEST_F(GnaGetScaleFactorTest, validSF) {
    EXPECT_NO_THROW(GetScaleFactorAndCheck(100, 200, 300, 400));
}

TEST_F(GnaGetScaleFactorTest, invalidSF) {
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(0, 200, 300, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, 0, 300, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, 200, 0, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, 200, 300, 0));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(-100, 200, 300, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, -200, 300, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, 200, -300, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, 200, 300, -400));
    double inf = std::numeric_limits<float>::infinity();
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(inf, 200, 300, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, inf, 300, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, 200, inf, 400));
    EXPECT_ANY_THROW(GetScaleFactorAndCheck(100, 200, 300, inf));
}

}  // namespace
