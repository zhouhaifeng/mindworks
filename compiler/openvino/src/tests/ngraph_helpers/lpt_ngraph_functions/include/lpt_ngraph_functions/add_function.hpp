// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <low_precision/layer_transformation.hpp>

#include "elementwise_function.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/convolution.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class AddActualValues {
public:
    ngraph::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ngraph::element::Type precision2;
    std::vector<float> subtractValues2;
    std::vector<float> mutliplyValues2;
};

inline std::ostream& operator<<(std::ostream& out, const AddActualValues& values) {
    return out <<
        "_" << values.precision1 <<
        "_subtract" << values.subtractValues1.size() <<
        "_mutliply" << values.mutliplyValues1.size() <<
        "_" << values.precision2 <<
        "_subtract" << values.subtractValues2.size() <<
        "_mutliply" << values.mutliplyValues2.size();
}

class AddExpectedValues {
public:
    ngraph::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ngraph::element::Type precision2;
    std::vector<float> mutliplyValuesAfter;
};

inline std::ostream& operator<<(std::ostream& out, const AddExpectedValues& values) {
    return out <<
        "_" << values.precision1 <<
        "_subtract" << values.subtractValues1.size() <<
        "_mutliply" << values.mutliplyValues1.size() <<
        "_" << values.precision2 <<
        "_mutliply" << values.mutliplyValuesAfter.size();
}

class AddFunction : public ElementwiseFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape1,
        const ngraph::PartialShape& inputShape2,
        const bool broadcast,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::element::Type& precision1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
        const ngraph::element::Type& precision2,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
        const int constInput,
        const std::vector<float>& constValues,
        const std::string& additionalLayer,
        const std::string& postops_configuration = "");

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool broadcast,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape1,
        const ngraph::PartialShape& inputShape2,
        const bool broadcast,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::element::Type& precision1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
        const ngraph::element::Type& precision2,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const int constInput,
        const std::vector<float>& constValues,
        const std::string& additionalLayer,
        const std::string& operationType,
        const std::string& postops_configuration = "");
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
