// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <low_precision/avg_pool.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/max_pool.hpp>
#include <low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp>
#include <memory>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "layer_transformation.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/markup_avg_pool_precisions_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph::pass;

OPENVINO_SUPPRESS_DEPRECATED_START

class MarkupAvgPoolPrecisionsTransformationTestValues {
public:
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ngraph::element::Type,
                   ngraph::Shape,
                   bool,         // additional FakeQuantize After
                   std::string,  // additional layer before FQ
                   MarkupAvgPoolPrecisionsTransformationTestValues>
    MarkupAvgPoolPrecisionsTransformationParams;

class MarkupAvgPoolPrecisionsTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<MarkupAvgPoolPrecisionsTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        MarkupAvgPoolPrecisionsTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = GetParam();

        actualFunction =
            ngraph::builder::subgraph::MarkupAvgPoolPrecisionsFunction::getOriginal(precision,
                                                                                    testValues.actual.inputPrecision,
                                                                                    shape,
                                                                                    addFakeQuantize,
                                                                                    additionalLayer,
                                                                                    testValues.actual.dequantization,
                                                                                    1,
                                                                                    0);

        ngraph::pass::low_precision::TypeRelaxedReplacer pass;
        pass.run_on_model(actualFunction);

        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::PrecisionsRestriction>(
            {ngraph::pass::low_precision::PrecisionsRestriction::create<ov::opset1::Convolution>(
                {{{0}, {ngraph::element::u8}}, {{1}, {ngraph::element::i8}}})});

        SimpleLowPrecisionTransformer transform(supportedPrecisionsOnActivation);
        transform.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::AvgPoolTransformation>();
        transform.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>();
        transform.commonGraphRewrite
            ->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
        transform.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>();
        transform.cleanup->add_matcher<ngraph::pass::low_precision::FakeQuantizeTransformation>();
        transform.cleanup->add_matcher<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
        transform.cleanup->add_matcher<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MarkupAvgPoolPrecisionsFunction::getReference(
            precision,
            testValues.expected.inputPrecision,
            shape,
            addFakeQuantize,
            additionalLayer,
            testValues.expected.dequantizationBefore,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MarkupAvgPoolPrecisionsTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool addFakeQuantize;
        std::string additionalLayer;
        MarkupAvgPoolPrecisionsTransformationTestValues testValues;
        std::tie(precision, shape, addFakeQuantize, additionalLayer, testValues) = obj.param;

        std::ostringstream result;
        result << precision << "_"
               << LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision,
                                                               shape,
                                                               testValues.params)
               << "_" << testValues.actual.dequantization << "_" << testValues.expected.dequantizationBefore << "_"
               << testValues.expected.preicsionAfterOperation << "_" << testValues.expected.dequantizationAfter << "_"
               << (addFakeQuantize ? "_FQ_after_" : "_") << additionalLayer;
        return result.str();
    }
};

TEST_P(MarkupAvgPoolPrecisionsTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    const auto avgPoolOperations = LayerTransformation::get<ov::op::v1::AvgPool>(actualFunction);
    ASSERT_EQ(1ul, avgPoolOperations.size()) << "unexpected avgPoolOperations size: " << avgPoolOperations.size();

    {
        auto avgPoolPrecisioinPreservedAttribute =
            ngraph::pass::low_precision::getAttribute<AvgPoolPrecisionPreservedAttribute>(*avgPoolOperations.begin());
        ASSERT_FALSE(avgPoolPrecisioinPreservedAttribute.empty());
        ASSERT_EQ(true, avgPoolPrecisioinPreservedAttribute.as<AvgPoolPrecisionPreservedAttribute>().value());
    }

    const auto precisionPreserved = LayerTransformation::get<ov::op::v1::MaxPool>(actualFunction);
    ASSERT_TRUE(checkIfAttributesAreTheSame<AvgPoolPrecisionPreservedAttribute>(precisionPreserved))
        << "AvgPoolPrecisionPreservedAttribute are not the same";

    // auto res = compare_functions(actualFunction, referenceFunction, true, true);
    // ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<std::string> additionalLayer = {
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {
    // true,
    false};

const std::vector<ngraph::Shape> shapes = {{1, 3, 9, 9}};

const std::vector<MarkupAvgPoolPrecisionsTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {LayerTransformation::createParamsU8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {128.f}, {0.02f}}},
     {ngraph::element::f32, {}, ngraph::element::f32, {{}, {128.f}, {0.02f}}}},
    // U8 without subtract
    {LayerTransformation::createParamsU8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {}, {0.02f}}},
     {ngraph::element::f32, {}, ngraph::element::f32, {{}, {}, {0.02f}}}},
    // U8 per channel quantization with different values
    {LayerTransformation::createParamsU8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {{128.f, 0.f, 128.f / 2}}, {{3.f, 1.f, 2.f}}}},
     {
         ngraph::element::f32,
         {{}, {}, {}},
         ngraph::element::f32,
         {{}, {{128.f, 0.f, 128.f / 2}}, {{3.f, 1.f, 2.f}}},
     }},
    // U8 per channel quantization with the same values
    {LayerTransformation::createParamsU8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {{128.f, 128.f, 128.f}}, {{3.f, 3.f, 3.f}}}},
     {
         ngraph::element::f32,
         {{}, {}, {}},
         ngraph::element::f32,
         {{}, {{128.f, 128.f, 128.f}}, {{3.f, 3.f, 3.f}}},
     }},
    // U8 without dequantization
    {LayerTransformation::createParamsU8I8(),
     {ngraph::element::f32, {}},
     {ngraph::element::f32, {}, ngraph::element::f32, {}}},
    // U8 not update precisions
    {LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
     {ngraph::element::f32, {{}, {128.f}, {0.02f}}},
     {ngraph::element::f32, {}, ngraph::element::f32, {{}, {128.f}, {0.02f}}}},
    // I8 per tensor quantization
    {LayerTransformation::createParamsI8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {128.f}, {0.02f}}},
     {ngraph::element::f32, {}, ngraph::element::f32, {{}, {128.f}, {0.02f}}}},
    // failed
    // I8 without subtract
    {LayerTransformation::createParamsI8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {}, {0.02f}}},
     {ngraph::element::f32, {}, ngraph::element::f32, {{}, {}, {0.02f}}}},
    // I8 per channel quantization with different values
    {LayerTransformation::createParamsI8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {{64.f, 0.f, 32.f}}, {{3.f, 1.f, 2.f}}}},
     {
         ngraph::element::f32,
         {{}, {}, {}},
         ngraph::element::f32,
         {{}, {{64.f, 0.f, 32.f}}, {{3.f, 1.f, 2.f}}},
     }},
    // I8 per channel quantization with the same values
    {LayerTransformation::createParamsI8I8(),
     {ngraph::element::f32, {{ngraph::element::f32}, {{64.f, 64.f, 64.f}}, {{3.f, 3.f, 3.f}}}},
     {
         ngraph::element::f32,
         {{}, {}, {}},
         ngraph::element::f32,
         {{}, {{64.f, 64.f, 64.f}}, {{3.f, 3.f, 3.f}}},
     }},
    // I8 without dequantization
    {LayerTransformation::createParamsI8I8(),
     {ngraph::element::f32, {}},
     {ngraph::element::f32, {}, ngraph::element::f32, {}}},
    // I8 not update precisions
    {LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
     {ngraph::element::f32, {{}, {128.f}, {0.02f}}},
     {ngraph::element::f32, {}, ngraph::element::f32, {{}, {128.f}, {0.02f}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         MarkupAvgPoolPrecisionsTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(addFQ),
                                            ::testing::ValuesIn(additionalLayer),
                                            ::testing::ValuesIn(testValues)),
                         MarkupAvgPoolPrecisionsTransformation::getTestCaseName);
