// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class PadTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::vector<uint64_t> padsBegin;
    std::vector<uint64_t> padsEnd;
    float padValue;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    ngraph::op::PadMode,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    PadTransformationParam
> PadTransformationParams;

class PadTransformation :
    public testing::WithParamInterface<PadTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PadTransformationParams>& obj);

protected:
    void SetUp() override;
    void Run() override;
};
}  // namespace LayerTestsDefinitions
