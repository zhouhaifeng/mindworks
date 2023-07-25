// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class VariadicSplitTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    int64_t splitedAxis;
    std::vector<size_t> splitLengths;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    VariadicSplitTransformationParam
> VariadicSplitTransformationParams;

class VariadicSplitTransformation :
    public testing::WithParamInterface<VariadicSplitTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<VariadicSplitTransformationParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
