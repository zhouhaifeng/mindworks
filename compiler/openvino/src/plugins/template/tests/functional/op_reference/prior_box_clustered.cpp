// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/prior_box.hpp"
#include "base_reference_test.hpp"
#include "openvino/opsets/opset1.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct PriorBoxClusteredParams {
    template <class IT>
    PriorBoxClusteredParams(const std::vector<float>& widths,
                            const std::vector<float>& heights,
                            const bool clip,
                            const ov::Shape& layerShapeShape, const ov::Shape& imageShapeShape,
                            const ov::element::Type& iType,
                            const std::vector<IT>& layerShapeValues, const std::vector<IT>& imageShapeValues,
                            const std::vector<float>& oValues,
                            const std::vector<float>& variances = {},
                            const std::string& testcaseName = "")
        : layerShapeShape(layerShapeShape),
          imageShapeShape(imageShapeShape),
          inType(iType),
          outType(ov::element::Type_t::f32),
          layerShapeData(CreateTensor(iType, layerShapeValues)),
          imageShapeData(CreateTensor(iType, imageShapeValues)),
          refData(CreateTensor(outType, oValues)),
          testcaseName(testcaseName) {
              attrs.widths = widths;
              attrs.heights = heights;
              attrs.clip = clip;
              if ( variances.size() != 0)
                attrs.variances = variances;
          }

    ov::op::v0::PriorBoxClustered::Attributes attrs;
    ov::Shape layerShapeShape;
    ov::Shape imageShapeShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor layerShapeData;
    ov::Tensor imageShapeData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferencePriorBoxClusteredLayerTest : public testing::TestWithParam<PriorBoxClusteredParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<PriorBoxClusteredParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "layerShapeShape=" << param.layerShapeShape << "_";
        result << "imageShapeShape=" << param.imageShapeShape << "_";
        result << "variancesSize=" << param.attrs.variances.size() << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PriorBoxClusteredParams& params) {
        auto LS = std::make_shared<opset1::Constant>(params.inType, params.layerShapeShape, params.layerShapeData.data());
        auto IS = std::make_shared<opset1::Constant>(params.inType, params.imageShapeShape, params.imageShapeData.data());
        const auto PriorBoxClustered = std::make_shared<op::v0::PriorBoxClustered>(LS, IS, params.attrs);
        return std::make_shared<ov::Model>(NodeVector {PriorBoxClustered}, ParameterVector {});
    }
};

TEST_P(ReferencePriorBoxClusteredLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<PriorBoxClusteredParams> generatePriorBoxClusteredFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PriorBoxClusteredParams> priorBoxClusteredParams {
        PriorBoxClusteredParams({3.0f}, {3.0f}, true,
                       {2}, {2},
                       IN_ET,
                       std::vector<T>{2, 2},
                       std::vector<T>{10, 10},
                       std::vector<float>{0,        0,        0.15f,    0.15f,    0.34999f, 0,    0.64999f, 0.15f, 0,    0.34999f, 0.15f,
                                          0.64999f, 0.34999f, 0.34999f, 0.64999f, 0.64999f, 0.1f, 0.1f,     0.1f,  0.1f, 0.1f,     0.1f,
                                          0.1f,     0.1f,     0.1f,     0.1f,     0.1f,     0.1f, 0.1f,     0.1f,  0.1f, 0.1f}),
        PriorBoxClusteredParams({3.0f}, {3.0f}, true,
                       {2}, {2},
                       IN_ET,
                       std::vector<T>{2, 2},
                       std::vector<T>{10, 10},
                       std::vector<float>{0,        0,        0.15f,    0.15f,    0.34999f, 0,    0.64999f, 0.15f, 0,    0.34999f, 0.15f,
                                          0.64999f, 0.34999f, 0.34999f, 0.64999f, 0.64999f, 0.1f, 0.2f,     0.3f,  0.4f, 0.1f,     0.2f,
                                          0.3f,     0.4f,     0.1f,     0.2f,     0.3f,     0.4f, 0.1f,     0.2f,  0.3f, 0.4f},
                       {0.1f, 0.2f, 0.3f, 0.4f}),
    };
    return priorBoxClusteredParams;
}

std::vector<PriorBoxClusteredParams> generatePriorBoxClusteredCombinedParams() {
    const std::vector<std::vector<PriorBoxClusteredParams>> priorBoxClusteredTypeParams {
        generatePriorBoxClusteredFloatParams<element::Type_t::i64>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::i32>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::i16>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::i8>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u64>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u32>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u16>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u8>(),
        };
    std::vector<PriorBoxClusteredParams> combinedParams;

    for (const auto& params : priorBoxClusteredTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_PriorBoxClustered_With_Hardcoded_Refs, ReferencePriorBoxClusteredLayerTest,
    testing::ValuesIn(generatePriorBoxClusteredCombinedParams()), ReferencePriorBoxClusteredLayerTest::getTestCaseName);

} // namespace
