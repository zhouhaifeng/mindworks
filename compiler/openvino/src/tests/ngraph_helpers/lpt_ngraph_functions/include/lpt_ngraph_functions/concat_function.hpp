// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConcatFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type inputPrecision,
        const ov::element::Type deqPrecision,
        const std::vector<ov::PartialShape>& inputShapes,
        const std::vector<DequantizationOperations>& dequantizationsBefore,
        const std::int64_t concatAxis,
        const ov::element::Type precisionAfter = ov::element::undefined,
        const DequantizationOperations& dequantizationAfter = {});

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const std::shared_ptr<ov::opset1::Constant>& input_constant1,
        const FakeQuantizeOnData& fakeQuantize1,
        const DequantizationOperations& dequantization1,
        const std::shared_ptr<ov::opset1::Constant>& input_constant2,
        const FakeQuantizeOnData& fakeQuantize2,
        const DequantizationOperations& dequantization2);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2);

    static std::shared_ptr<ngraph::Function> getOriginalWithChildAndOutput(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2);

    static std::shared_ptr<ngraph::Function> getOriginalWithNeighbors(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3,
        const std::string& neighborType,
        const std::string& additionalLayer);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediateAvgPool(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithSplitedIntermediate(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const bool addConvolution);

    static std::shared_ptr<ngraph::Function> getOriginalSelectionWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithStridedSlice(
        const ngraph::element::Type precision,
        const ngraph::PartialShape inputShape,
        const FakeQuantizeOnData& fq1,
        const FakeQuantizeOnData& fq2,
        const bool ssBeforeConcat,
        const bool ssAfterConcat);

    static std::shared_ptr<ngraph::Function> getOriginalWithDifferentPrecisionOnChildren(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const std::int64_t axis,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediateWithConstant(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalWithReshapeAtTheEndTransformation(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData1,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const FakeQuantizeOnDataWithConstant& fqOnData3);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediateReshape(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& reshapeOutputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type dequantizationPrecision,
        const ov::element::Type precisionBefore,
        const std::vector<ov::PartialShape>& inputShapes,
        const std::vector<DequantizationOperations>& dequantizationsBefore,
        const ov::element::Type precisionAfter,
        const DequantizationOperations& dequantizationAfter,
        const std::int64_t concatAxis);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2,
        const DequantizationOperations& dequantizationOperations);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const DequantizationOperations::Convert& convert1,
        const DequantizationOperations& dequantization1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2,
        const DequantizationOperations::Convert& convert2,
        const DequantizationOperations& dequantization2,
        const std::vector<ov::Any>& concatAttributes,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const std::int64_t& axis,
        const bool addNotPrecisionPreservedOperation = false);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape& inputShape1,
        const FakeQuantizeOnDataWithConstant& fakeQuantize1,
        const DequantizationOperations::Convert& convert1,
        const DequantizationOperations& dequantization1,
        const bool addReshape1,
        const ngraph::Shape& inputShape2,
        const FakeQuantizeOnDataWithConstant& fakeQuantize2,
        const DequantizationOperations::Convert& convert2,
        const DequantizationOperations& dequantization2,
        const bool addReshape2,
        const std::vector<ov::Any>& concatAttributes,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const std::int64_t& axis,
        const bool addNotPrecisionPreservedOperation = false);

    static std::shared_ptr<ngraph::Function> getReferenceWithNeighbors(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2,
        const std::string& neighborType,
        const std::string& additionalLayer);

    // TODO: refactor: dequantizationBefore2 <=> dequantizationOperations2
    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationOperations2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationBefore2);

    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediateAvgPool(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceWithSplitedIntermediate(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ngraph::element::Type precisionAfterOperation,
        const bool addConvolution,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceSelectionWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceWithStridedSlice(
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape inputShape,
        const FakeQuantizeOnData& fq1,
        const FakeQuantizeOnData& fq2,
        const DequantizationOperations& deqBefore,
        const ngraph::element::Type precisionBeforeConcat,
        const ngraph::element::Type precisionAfterConcat,
        const bool ssBeforeConcat,
        const bool ssAfterConcat,
        const DequantizationOperations& deqAfter1,
        const DequantizationOperations& deqAfter2);

    static std::shared_ptr<ngraph::Function> getReferenceWithDifferentPrecisionOnChildren(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool multiChannel,
        const std::int64_t axis,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore1,
        const DequantizationOperations& dequantizationBefore2,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter1,
        const DequantizationOperations& dequantizationAfter2);

    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediateWithConstant(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const ngraph::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const ngraph::element::Type precisionAfterDequantization);

    static std::shared_ptr<ngraph::Function> getReferenceWithReshapeAtTheEndTransformation(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData1,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const FakeQuantizeOnDataWithConstant& fqOnData3,
        const ngraph::element::Type precisionBeforeOp,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations);

    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediateReshape(
            const ngraph::element::Type precision,
            const ngraph::Shape& inputShape,
            const ngraph::Shape& reshapeOutputShape,
            const FakeQuantizeOnData& fqOnData1,
            const FakeQuantizeOnData& fqOnData2,
            const DequantizationOperations& dequantizationAfter);

private:
    static std::shared_ptr<Node> makeMaxPool(const Output<Node>& parent, const std::vector<size_t>& kernel);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
