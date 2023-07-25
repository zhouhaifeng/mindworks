// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "base_reference_test.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct MultiplyParams {
    template <class IT>
    MultiplyParams(const PartialShape& iShape1,
              const PartialShape& iShape2,
              const element::Type& iType,
              const std::vector<IT>& iValues1,
              const std::vector<IT>& iValues2,
              const std::vector<IT>& oValues)
        : pshape1(iShape1),
          pshape2(iShape2),
          inType(iType),
          outType(iType),
          inputData1(CreateTensor(iType, iValues1)),
          inputData2(CreateTensor(iType, iValues2)),
          refData(CreateTensor(iType, oValues)) {}

    PartialShape pshape1;
    PartialShape pshape2;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

class ReferenceMultiplyLayerTest : public testing::TestWithParam<MultiplyParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape1=" << param.pshape1 << "_";
        result << "iShape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape1,
                                                    const PartialShape& input_shape2,
                                                    const element::Type& input_type,
                                                    const element::Type& expected_output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        const auto multiply = std::make_shared<op::v1::Multiply>(in1, in2);

        return std::make_shared<Model>(NodeVector{multiply}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceMultiplyLayerTest, MultiplyWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<MultiplyParams> generateParamsForMultiply() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<MultiplyParams> params{
        MultiplyParams(ov::PartialShape{2, 2},
                       ov::PartialShape{2, 2},
                       IN_ET,
                       std::vector<T>{1, 2, 3, 4},
                       std::vector<T>{5, 6, 7, 8},
                       std::vector<T>{5, 12, 21, 32}),
        MultiplyParams(ov::PartialShape{3, 2, 1},
                       ov::PartialShape{1, 6},
                       IN_ET,
                       std::vector<T>{12, 24, 36, 48, 60, 72},
                       std::vector<T>{1, 2, 3, 4, 6, 1},
                       std::vector<T>{12, 24,  36,  48,  72,  12, 24, 48,  72,  96,  144, 24,
                                      36, 72,  108, 144, 216, 36, 48, 96,  144, 192, 288, 48,
                                      60, 120, 180, 240, 360, 60, 72, 144, 216, 288, 432, 72}),
        MultiplyParams(ov::PartialShape{1},
                       ov::PartialShape{1},
                       IN_ET,
                       std::vector<T>{2},
                       std::vector<T>{8},
                       std::vector<T>{16})
    };
    return params;
}

template <element::Type_t IN_ET>
std::vector<MultiplyParams> generateParamsForMultiplyFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<MultiplyParams> params{
        MultiplyParams(ov::PartialShape{1},
                       ov::PartialShape{1},
                       IN_ET,
                       std::vector<T>{3.1},
                       std::vector<T>{8},
                       std::vector<T>{24.8})
    };
    return params;
}

std::vector<MultiplyParams> generateCombinedParamsForMultiply() {
    const std::vector<std::vector<MultiplyParams>> allTypeParams{
        generateParamsForMultiply<element::Type_t::f32>(),
        generateParamsForMultiply<element::Type_t::f16>(),
        generateParamsForMultiply<element::Type_t::bf16>(),
        generateParamsForMultiply<element::Type_t::i64>(),
        generateParamsForMultiply<element::Type_t::i32>(),
        generateParamsForMultiply<element::Type_t::u64>(),
        generateParamsForMultiply<element::Type_t::u32>()
    };

    std::vector<MultiplyParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<MultiplyParams> generateCombinedParamsForMultiplyFloat() {
    const std::vector<std::vector<MultiplyParams>> allTypeParams{
        generateParamsForMultiplyFloat<element::Type_t::f32>(),
        generateParamsForMultiplyFloat<element::Type_t::f16>()
    };

    std::vector<MultiplyParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Multiply_With_Hardcoded_Refs,
    ReferenceMultiplyLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForMultiply()),
    ReferenceMultiplyLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Multiply_Float_With_Hardcoded_Refs,
    ReferenceMultiplyLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForMultiplyFloat()),
    ReferenceMultiplyLayerTest::getTestCaseName);

}  // namespace
