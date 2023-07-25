// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"

#include "low_precision/move_fake_quantize.hpp"

#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"

namespace LayerTestsDefinitions {

class MoveFakeQuantizeTransformationParam {
public:
    size_t concatInputsCount;
    std::string operation;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeAfter;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertAfter;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    std::string layerName;
    std::string expectedKernelType;
    std::int64_t axis;
};

typedef std::tuple <
    ngraph::element::Type,
    std::vector<ngraph::PartialShape>,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    bool,
    MoveFakeQuantizeTransformationParam
> MoveFakeQuantizeTransformationParams;

class MoveFakeQuantizeTransformation :
    public testing::WithParamInterface<MoveFakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MoveFakeQuantizeTransformationParams> obj);

protected:
    void SetUp() override;

    void Run() override;
};

}  // namespace LayerTestsDefinitions
