// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mat_mul.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include <string>
#include <ov_ops/type_relaxed.hpp>
#include "shared_test_classes/base/utils/generate_inputs.hpp"
#include "cpu/cpu_config.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        ElementType,                        // Input precision
        ElementType,                        // Weights precision
        ElementType,                        // Output precision
        fusingSpecificParams,
        CPUSpecificParams,
        std::map<std::string, std::string>, // Additional config
        float                               // Weights sparse rate
> MatMulSparseParamSet;

class MatMulSparseCPUTest : public testing::WithParamInterface<MatMulSparseParamSet>,
                            virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulSparseParamSet>& obj) {
        ShapeRelatedParams shapeRelatedParams;
        ElementType inType, weiType, outType;
        fusingSpecificParams fusingParams;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        float weiSparseRate;
        std::tie(shapeRelatedParams, inType, weiType, outType, fusingParams, cpuParams, additionalConfig,
            weiSparseRate) = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << CommonTestUtils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << shapeRelatedParams.transpose.first << "_";
        result << "transpose_b=" << shapeRelatedParams.transpose.second << "_";
        result << "inType=" << inType << "_";
        result << "weiType=" << weiType << "_";
        result << "outType=" << outType << "_";
        result << CpuTestWithFusing::getTestCaseName(fusingParams);
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }
        result << "_weiSparseRate=" << weiSparseRate;

        return result.str();
    }

protected:
     std::string cpuNodeType;

    template<typename T>
    void transpose(T& shape) {
        IE_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    std::vector<int8_t> inline generateSparseVector(size_t vec_len,
                float sparseRate = 0.0f,
                int8_t upTo = 10,
                int8_t startFrom = 1,
                int32_t seed = 1) {
        std::vector<int8_t> res(vec_len);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<long> dist(static_cast<long>(startFrom), static_cast<long>(upTo));

        std::mt19937 gen_f(123);
        std::uniform_real_distribution<float> dist_f(0.f, 1.f);

        int countZero = 0;

        res[0] = startFrom;
        res[vec_len - 1] = upTo;
        for (size_t i = 1; i < vec_len - 1; i++) {
            if (dist_f(gen_f) > sparseRate) {
                res[i] = static_cast<int8_t>(dist(gen));
            } else {
                res[i] = 0;
                countZero++;
            }
        }

        std::cout << "Sparse rate = " << countZero * 100 / vec_len << "%" << std::endl;

        return res;
    }

    std::shared_ptr<Node> makeMatMulRelaxed(const Output<Node>& A,
                                            const ov::PartialShape& inShapeB,
                                            ElementType weiType,
                                            bool transpose_a,
                                            bool transpose_b,
                                            const std::vector<int8_t>& weiData) {
        using namespace ngraph;
        auto inputParamsFP32 = builder::makeDynamicParams(element::f32, {A.get_partial_shape()});
        auto matrixBFP32 = builder::makeDynamicInputLayer(element::f32, helpers::InputLayerType::CONSTANT, inShapeB);

        auto matMulRelaxed = std::make_shared<ov::op::TypeRelaxed<opset3::MatMul>>(
            *as_type_ptr<opset3::MatMul>(builder::makeMatMul(inputParamsFP32[0], matrixBFP32, transpose_a, transpose_b)),
            element::f32);

        auto matrixB = ngraph::builder::makeConstant<int8_t>(weiType, inShapeB.get_shape(), weiData);

        auto matMul = matMulRelaxed->copy_with_new_inputs({A, matrixB});

        return matMul;
    }

    void SetUp() override {
        abs_threshold = 0.5f;
        using ngraph::pass::ConvertPrecision;

        ShapeRelatedParams shapeRelatedParams;
        ElementType inType, weiType, outType;
        fusingSpecificParams fusingParams;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        float weiSparseRate;

        std::tie(shapeRelatedParams, inType, weiType, outType, fusingParams, cpuParams, additionalConfig,
            weiSparseRate) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        targetDevice = CommonTestUtils::DEVICE_CPU;

        init_input_shapes(shapeRelatedParams.inputShapes);

        bool transpA = shapeRelatedParams.transpose.first;
        bool transpB = shapeRelatedParams.transpose.second;

        if (transpA) {
            transpose(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[0]);
            }
        }
        if (transpB) {
            transpose(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[1]);
            }
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        cpuNodeType = "FullyConnected";
        selectedType = makeSelectedTypeStr(selectedType, element::i8);

        auto params = builder::makeDynamicParams(inType, {inShapeA});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));

        auto matrixB = builder::makeDynamicInputLayer(element::f32, helpers::InputLayerType::CONSTANT, inShapeB);

        auto weiData = generateSparseVector(ngraph::shape_size(inShapeB.get_shape()), weiSparseRate);
        auto matMul = makeMatMulRelaxed(paramOuts[0], inShapeB, weiType, transpA, transpB, weiData);

        function = makeNgraphFunction(element::f32, params, matMul, cpuNodeType);

        checkFusingPosition = false;

        functionRefs = ov::clone_model(*function);
        ngraph::pass::ConvertPrecision<ngraph::element::Type_t::i8, ngraph::element::Type_t::f32>().run_on_model(functionRefs);
        ngraph::pass::ConvertPrecision<ngraph::element::Type_t::u8, ngraph::element::Type_t::f32>().run_on_model(functionRefs);
        functionRefs->validate_nodes_and_infer_types();
    }
};

TEST_P(MatMulSparseCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, cpuNodeType);
}

namespace {

/* ============= Common params ============= */

std::vector<CPUSpecificParams> filterSpecificParams(bool sparseExpected) {
    std::vector<CPUSpecificParams> specificParams;
    if (with_cpu_x86_avx512_core_amx()) {
        if (sparseExpected) {
            specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx512_amx"}, "brgemm_avx512_amx_sparse"});
        } else {
            specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx512_amx"}, "brgemm_avx512_amx"});
        }
    }

    return specificParams;
}

/* ============= FullyConnected ============= */
namespace fullyConnected {

// cpu (sparse) configs
const std::map<std::string, std::string> emptyConfig = {};
const std::map<std::string, std::string> SparseRate50 = {{CPUConfigParams::KEY_CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE, "0.5"}};
const std::map<std::string, std::string> SparseRate80 = {{CPUConfigParams::KEY_CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE, "0.8"}};

const std::vector<ShapeRelatedParams> IS2D_sparse_smoke = {
    {static_shapes_to_test_representation({{64, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{71, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{3, 128}, {128, 64}}), {false, true}},
    {static_shapes_to_test_representation({{71, 64}, {64, 128}}), {false, true}},

    {
        {
            {{-1, -1}, {{20, 64}, {20, 64}}},
            {{64, 128}, {{64, 128}, {64, 128}}}
        },
        {false, true}
    },

    {
        {
            {{{0, 100}, {0, 64}}, {{20, 64}, {14, 64}, {20, 64}, {14, 64}}},
            {{64, 128}, {{64, 128}, {64, 128}, {64, 128}, {64, 128}}}
        },
        {false, true}
    },
};

const auto testParams2D_i8_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(false)),
                                                   ::testing::Values(emptyConfig, SparseRate80),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_I8, MatMulSparseCPUTest, testParams2D_i8_smoke,
    MatMulSparseCPUTest::getTestCaseName);

const auto testParams2D_i8_sparse_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(true)),
                                                   ::testing::Values(SparseRate50),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_I8_sparse, MatMulSparseCPUTest, testParams2D_i8_sparse_smoke,
    MatMulSparseCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS3D_sparse_smoke = {
    {static_shapes_to_test_representation({{1, 64, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{3, 71, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{3, 5, 128}, {128, 64}}), {false, true}},
    {static_shapes_to_test_representation({{1, 71, 64}, {64, 128}}), {false, true}},

    {
        {
            {{-1, -1, 64}, {{1, 5, 64}, {1, 10, 64}, {1, 5, 64}, {1, 10, 64}}},
            {{64, 128}, {{64, 128}, {64, 128}}}
        },
        {false, true}
    },

    // todo: [av] investigate "Primitive descriptor was not found" error for this case
    // {
    //     {
    //         {{{0, 60}, {0, 60}, {0, 64}}}, {{1, 3, 64}, {1, 7, 64}}},
    //         {{64, 64}, {{64, 64}, {64, 64}}}
    //     },
    //     {false, true}
    // },
};

const auto testParams3D_i8_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(false)),
                                                   ::testing::Values(emptyConfig, SparseRate80),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_I8, MatMulSparseCPUTest, testParams3D_i8_smoke,
    MatMulSparseCPUTest::getTestCaseName);

const auto testParams3D_i8_sparse_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(true)),
                                                   ::testing::Values(SparseRate50),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_I8_sparse, MatMulSparseCPUTest, testParams3D_i8_sparse_smoke,
    MatMulSparseCPUTest::getTestCaseName);

} // namespace fullyConnected

} // namespace

} // namespace CPULayerTestsDefinitions
