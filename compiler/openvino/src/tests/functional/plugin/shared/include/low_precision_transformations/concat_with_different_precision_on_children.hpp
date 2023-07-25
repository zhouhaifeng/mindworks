// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class ConcatWithDifferentChildrenTransformationParam {
public:
    std::int64_t axis;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string, // target device: CPU, GPU
    ConcatWithDifferentChildrenTransformationParam,
    ngraph::pass::low_precision::LayerTransformation::Params // transformation parameters
    > ConcatWithDifferentChildrenTransformationParams;

class ConcatWithDifferentChildrenTransformation :
    public testing::WithParamInterface<ConcatWithDifferentChildrenTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatWithDifferentChildrenTransformationParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
