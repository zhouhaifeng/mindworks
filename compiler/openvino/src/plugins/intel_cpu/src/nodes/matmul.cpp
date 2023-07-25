// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.h"

#include "ie_precision.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "eltwise.h"

#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "fake_quantize.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <dnnl_extension_utils.h>
#include <common/primitive_hashing_utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct MatMulKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    dnnl::primitive_attr attr;
    impl_desc_type implType;

    size_t hash() const;
    bool operator==(const MatMulKey& rhs) const;
};

size_t MatMulKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    return seed;
}

bool MatMulKey::operator==(const MatMulKey &rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get() &&
             implType == rhs.implType;
    return retVal;
}
} // namespace

bool MatMul::canBeExecutedInInt8() const {
    auto firstInputPrecision = getOriginalInputPrecisionAtPort(0);
    auto secondInputPrecision = getOriginalInputPrecisionAtPort(1);

    return one_of(firstInputPrecision, Precision::U8, Precision::I8) && secondInputPrecision == Precision::I8;
}

bool MatMul::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
class MMShapeInfer : public ShapeInferEmptyPads {
public:
    MMShapeInfer(const size_t& out_rank, const bool& transpose_a, const bool& transpose_b) :
        m_out_rank(out_rank), m_transpose_a(transpose_a), m_transpose_b(transpose_b) {
        m_shapeY = VectorDims(m_out_rank, 1); // for output and cache
    }
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const VectorDims& shapeA = input_shapes[0].get();
        const VectorDims& shapeB = input_shapes[1].get();
        const size_t rankA = shapeA.size();
        const size_t rankB = shapeB.size();

        // getSupportedDescriptors has done some shape check.
        // 1. Needn't assert the scalar type since the matmul_shape_inference has checked.
        // 2. Needn't check the compatibility of the last two dims
        // 3. 1-D x 1-D is needed
        // 4. transpose is necessary
        // 5. Just support the same rank of matmul
        // 6. simplify the broadcast check
        if (rankA == 1 && rankB == 1 && shapeA[0] == shapeB[0]) {
            return {{m_shapeY}, ShapeInferStatus::success};
        }

        m_shapeY[m_out_rank-2] = m_transpose_a ? shapeA[rankA-1] : shapeA[rankA-2];
        m_shapeY[m_out_rank-1] = m_transpose_b ? shapeB[rankB-2] : shapeB[rankB-1];

        for (size_t i=0; i < m_out_rank-2; ++i) {
            if (shapeA[i] != shapeB[i]) {
                if (shapeB[i] == 1) {
                    m_shapeY[i] = shapeA[i];
                    continue;
                } else if (shapeA[i] != 1) {
                    IE_THROW() << "Incompatible MatMul batch dimension. Cant merge the first input dimension=" <<
                                  shapeA[i] << " with second input dimension=" << shapeB[i] << " at index=" << i;
                }
            }
            m_shapeY[i] = shapeB[i];
        }

        return {{m_shapeY}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    VectorDims m_shapeY;
    const size_t m_out_rank;
    const bool m_transpose_a;
    const bool m_transpose_b;
};

class MMShapeInferFactory : public ShapeInferFactory {
public:
    MMShapeInferFactory(const std::shared_ptr<ngraph::Node>& op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        if (const auto matmul = ov::as_type_ptr<const ngraph::opset1::MatMul>(m_op)) {
            const auto output_rank = matmul->get_output_partial_shape(0).rank().get_length();
            const bool transpose_a = matmul->get_transpose_a();
            const bool transpose_b = matmul->get_transpose_b();
            return std::make_shared<MMShapeInfer>(output_rank, transpose_a, transpose_b);
       } else {
             IE_THROW() << "Unexpected operation type in the MatMul shape inference factory";
       }
    }
private:
    std::shared_ptr<ngraph::Node> m_op;
};
} // namespace

MatMul::MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
    Node(op, context, MMShapeInferFactory(op)), withBiases(false) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    if (!matMul) {
        IE_THROW(NotImplemented) << "Operation with name " << op->get_friendly_name() << ":" << op->get_type_name() <<
            " is not an instance of MatMul from opset1";
    }

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();
}

bool MatMul::canFuse(const NodePtr& node) const {
    // WA for CVS-84056: oneDNN brgemm impl has problem with per-OC binary-postOps for MatMul with 6D inputs
    if (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            if (eltwiseNode->getBroadcastingPolicy() == Eltwise::BroadcastingPolicy::PerChannel) {
                auto rank = getInputShapeAtPort(0).getRank();
                if (rank > 4) {
                    DEBUG_LOG("skip fusing non-perTensor Eltwise:", eltwiseNode->getName(), " into 6D MatMul:", getName());
                    return false;
                }
            }
        }
    }

    //  Consider the case when Matmul doesn't support execution in int8, but is getting fused with FQ with int8 output.
    //  Then the Matmul will change its output precision to fp32. If fusing FQ into matmul, there would be reorder inserted
    //  after matmul. In some bert model, this reorder causes great perf degradation.
    //  Todo: Remove this if onednn primitive support U8 output with floating input.
    if (node->getType() == Type::FakeQuantize && one_of(node->getOriginalOutputPrecisionAtPort(0), Precision::I8, Precision::U8) &&
        !canBeExecutedInInt8() &&
        getOriginalInputPrecisionAtPort(0) == InferenceEngine::Precision::FP32 )
        return false;
    return canFuseSimpleOperation(node);
}

void MatMul::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims, bool initWeights = false) {
    dnnl::post_ops ops;

    dnnl::memory::data_type outputDataType = dnnl::memory::data_type::undef;
    if (outDataDesc) {
        outputDataType = outDataDesc->getDataType();
    }

    bool isINT8 = canBeExecutedInInt8();

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8, 1 << (dims.size() - 1), getDQScales(), withBiases);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

Node::AttrPtr MatMul::initPrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims, true);

    (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    return attr;
}

Node::AttrPtr MatMul::initPrimitiveAttr() {
    auto dummyShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
    return initPrimitiveAttr(dummyShape.getStaticDims());
}

/* Example MatMul:
 * 2x128x512(T) * 2x128x512 = 2x512x512
 * First input 2x128x512(T) should be transposed
 * oneDNN requires memory::desc for this input to:
 * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
 * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
 */
static VectorDims getStridesAndModifyShape(Shape& shape, const bool transpose) {
    const auto getRank = shape.getRank();

    VectorDims strides(getRank, 1);
    const auto& staticDims = shape.getStaticDims();
    for (size_t i = 1; i < getRank; i++) {
        strides[getRank - i - 1 ] = strides[getRank - i] * staticDims[getRank - i];
    }

    if (transpose && getRank > 1) {
        // form new shape
        auto dims = staticDims;
        std::swap(dims[getRank - 2], dims[getRank - 1]);
        shape = Shape{dims};
        // update strides
        strides[getRank - 1] = staticDims[getRank - 2];
        strides[getRank - 2] = 1;
    }

    return strides;
}

dnnl::memory::desc MatMul::getBiasDescFrom(const DnnlMemoryDescCPtr outMemDesc) {
    // oneDNN matmul requires shape for bias desc to be the same rank
    VectorDims biasDims(outMemDesc->getShape().getRank(), 1);
    const auto outDims = outMemDesc->getShape().getStaticDims();
    const auto chIdx = getFusingAxis();
    biasDims[chIdx] = outDims[chIdx];
    const auto bdt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(2));

    return dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(biasDims), bdt, memory::format_tag::any);
}

void MatMul::getSupportedDescriptors() {
    if (getParentEdges().size() != getOriginalInputsNumber())
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    withBiases = getOriginalInputsNumber() == 3;

    auto firstInPortPrec = getOriginalInputPrecisionAtPort(0);
    auto secondInPortPrec = getOriginalInputPrecisionAtPort(1);
    auto outPortPrec = getOriginalOutputPrecisionAtPort(0);

    if (firstInPortPrec.size() != secondInPortPrec.size())
        firstInPortPrec = secondInPortPrec = getMaxPrecision(getOriginalInputPrecisions());

    // fallback to fp32 for any precision that cannot be handled natively
    if ((!one_of(firstInPortPrec , Precision::U8, Precision::I8, Precision::BF16, Precision::FP16, Precision::FP32) ||
         !one_of(secondInPortPrec , Precision::I8, Precision::BF16, Precision::FP16, Precision::FP32))) {
        outPortPrec = firstInPortPrec = secondInPortPrec = Precision::FP32;
    }

    Precision postOpsPrec = outPortPrec;
    if (!fusedWith.empty()) {
        postOpsPrec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (canBeExecutedInInt8()) {
        // INT8 mode support wide range of output precisions
        outPortPrec = postOpsPrec;
        // INT8 matmul do not support fp16 output
        if (outPortPrec == Precision::FP16) {
            outPortPrec = Precision::FP32;
        }
    } else if (postOpsPrec == Precision::FP32) {
        // all non-INT8 modes support fp32 output precision
        outPortPrec = postOpsPrec;
    } else {
        // otherwise we ignore postOpsPrec and stay with getOriginalOutputPrecisionAtPort(0)
    }

    const auto& inputShape0 = getInputShapeAtPort(0);
    const auto& inputShape1 = getInputShapeAtPort(1);
    const auto& outputShape = getOutputShapeAtPort(0);

    if (inputShape0.getRank() != inputShape1.getRank() || inputShape0.getRank() != outputShape.getRank())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    const int nDims = inputShape0.getRank();
    const auto xAxis = nDims - 1;
    const auto yAxis = nDims - 2;
    const auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    const auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    const auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    const auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    const auto& inDims0 = getInputShapeAtPort(0).getDims();
    const auto& inDims1 = getInputShapeAtPort(1).getDims();
    const auto& outDims = getOutputShapeAtPort(0).getDims();

    // coverity[copy_paste_error]
    if (!dimsEqualWeak(inDims0[xAxis0], inDims1[yAxis1]) ||
        !dimsEqualWeak(inDims0[yAxis0], outDims[yAxis]) ||
        !dimsEqualWeak(inDims1[xAxis1], outDims[xAxis]))
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((!dimsEqualWeak(inDims0[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims0[dim_idx], 1)) ||
            (!dimsEqualWeak(inDims1[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims1[dim_idx], 1))) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    std::vector<Shape> staticInputShapes{inputShape0, inputShape1};
    if (inputShape0.isDynamic() || inputShape1.isDynamic()) {
        std::tie(staticInputShapes[0], staticInputShapes[1]) = makeDummyInputShapes(inputShape0, inputShape1);
    }

    auto staticOutputShape = outputShape.isStatic() ? outputShape : Shape(shapeInferGeneric(staticInputShapes).front());

    const VectorDims inStrides0 = getStridesAndModifyShape(staticInputShapes[0], transposeIn[0]);
    const VectorDims inStrides1 = getStridesAndModifyShape(staticInputShapes[1], transposeIn[1]);

    inDataDesc[0] = std::make_shared<DnnlBlockedMemoryDesc>(firstInPortPrec, staticInputShapes[0], inStrides0);
    inDataDesc[1] = std::make_shared<DnnlBlockedMemoryDesc>(secondInPortPrec, staticInputShapes[1], inStrides1);
    outDataDesc   = std::make_shared<DnnlBlockedMemoryDesc>(outPortPrec, staticOutputShape);

    createDescriptor({inDataDesc[0], inDataDesc[1]}, {outDataDesc});
}

std::pair<Shape, Shape> MatMul::makeDummyInputShapes(const Shape& in0, const Shape& in1) const {
    if (in0.getRank() < 2 || in1.getRank() < 2) {
        IE_THROW() << "Can't create dummy inputs with rank less 2";
    }

    if (in0.getRank() != in1.getRank()) {
        IE_THROW() << "Can't create dummy inputs if input's rank not equal";
    }

    auto swapTranspDims = [&](VectorDims& in0, VectorDims& in1) {
        if (transposeIn[0]) {
            std::swap(in0[in0.size() - 1], in0[in0.size() - 2]);
        }
        if (transposeIn[1]) {
            std::swap(in1[in1.size() - 1], in1[in1.size() - 2]);
        }
    };

    auto inDims0 = in0.getDims();
    auto inDims1 = in1.getDims();

    auto minDims0 = in0.getMinDims();
    auto maxDims0 = in0.getMaxDims();
    auto minDims1 = in1.getMinDims();
    auto maxDims1 = in1.getMaxDims();

    swapTranspDims(inDims0, inDims1);
    swapTranspDims(minDims0, minDims1);
    swapTranspDims(maxDims0, maxDims1);

    auto fillDummy = [&](size_t idx0, size_t idx1) {
        if (inDims0[idx0] == Shape::UNDEFINED_DIM && inDims1[idx1] == Shape::UNDEFINED_DIM) {
            inDims0[idx0] = inDims1[idx1] = std::min(std::min(maxDims0[idx0], maxDims1[idx1]),
                                            std::max(std::max(minDims0[idx0], minDims1[idx1]), static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
        } else {
            if (inDims0[idx0] == Shape::UNDEFINED_DIM && inDims1[idx1] != Shape::UNDEFINED_DIM) {
                if (inDims1[idx1] == 1 && minDims0[idx0] != Shape::UNDEFINED_DIM) {
                    inDims0[idx0] = std::max<Dim>(minDims0[idx0], 1);
                } else {
                    inDims0[idx0] = inDims1[idx1];
                }
            } else if (inDims0[idx0] != Shape::UNDEFINED_DIM && inDims1[idx1] == Shape::UNDEFINED_DIM) {
                if (inDims0[idx0] == 1 && minDims1[idx1] != Shape::UNDEFINED_DIM) {
                    inDims1[idx1] = std::max<Dim>(minDims1[idx1], 1);
                } else {
                    inDims1[idx1] = inDims0[idx0];
                }
            }
        }
    };

    // fill k
    fillDummy(inDims0.size() - 1, inDims1.size() - 2);

    // fill m, n
    if (inDims0[inDims0.size() - 2] == Shape::UNDEFINED_DIM) {
        inDims0[inDims0.size() - 2] = std::min(maxDims0[inDims0.size() - 2],
                                               std::max(minDims0[inDims0.size() - 2], static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
    }
    if (inDims1[inDims1.size() - 1] == Shape::UNDEFINED_DIM) {
        inDims1[inDims1.size() - 1] = std::min(maxDims1[inDims1.size() - 1],
                                               std::max(minDims1[inDims1.size() - 1], static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
    }

    // fill batches
    for (size_t i = 0; i < inDims0.size() - 2; i++) {
        fillDummy(i, i);
    }

    swapTranspDims(inDims0, inDims1);

    return {Shape(inDims0), Shape(inDims1)};
}

void MatMul::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                              const std::vector<MemoryDescPtr>& outputDesc) {
    const auto attr = initPrimitiveAttr();
    dnnl::matmul::primitive_desc matmul_desc;
    if (withBiases) {
        matmul_desc = matmul::primitive_desc(
            getEngine(),
            inDataDesc[0]->getDnnlDesc(),
            inDataDesc[1]->getDnnlDesc(),
            getBiasDescFrom(outDataDesc),
            outDataDesc->getDnnlDesc(),
            *attr);
    } else {
        matmul_desc = matmul::primitive_desc(
            getEngine(),
            inDataDesc[0]->getDnnlDesc(),
            inDataDesc[1]->getDnnlDesc(),
            outDataDesc->getDnnlDesc(),
            *attr);
    }

    descs.emplace_back(matmul_desc);
}

void MatMul::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = canBeInPlace() ? 0 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);

            inConfs.emplace_back(desc);
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);

            outConfs.emplace_back(desc, BlockedMemoryDesc::FULL_MASK, inPlaceOutPort);
        }

        const NodeConfig config(inConfs, outConfs);
        const impl_desc_type impl_type = parse_impl_name(prim_desc.impl_info_str());

        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    };

    for (auto& desc : descs) {
        auto first_desc = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(desc.get()));

        const bool first_match = customImplPriorities.empty();
        DnnlExtensionUtils::for_each_implementation(desc,
                                                    first_match,
                                                    [&](impl_desc_type implType) {
                                                        return contains(getImplPriority(), implType);
                                                    },
                                                    [&](dnnl::primitive_desc& desc) {
                                                        addSupportedPrimitiveDescriptor(desc);
                                                    });

        // fallback. if none of the primitive types is present in the priority list just add first implementation
        // @todo this fallback is not necessary if primitive priority list is filled correctly
        if (supportedPrimitiveDescriptors.empty())
            addSupportedPrimitiveDescriptor(first_desc);
   }
}

MemoryDescPtr MatMul::getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc = idx > 0 ? prim_desc.weights_desc(idx - 1): prim_desc.src_desc(idx);

    if (idx < 2) // inputs
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()),
            getInputShapeAtPort(idx)); /* provide initial shapes, so hide transpose effect */
    else // bias
        return DnnlExtensionUtils::makeDescriptor(desc);
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

InferenceEngine::Precision MatMul::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void MatMul::prepareParams() {
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->isAllocated() || !src1MemPtr || !src1MemPtr->isAllocated())
        IE_THROW()  << errorPrefix << " did not allocate input memory";

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    DnnlMemoryDescPtr src0TransposedDesc;
    DnnlMemoryDescPtr src1TransposedDesc;

    AttrPtr attr;

    if (isDynamicNode()) {
        attr = initPrimitiveAttr(dstMemPtr->getStaticDims());

        const auto& src0Desc = src0MemPtr->getDesc();
        const auto& src1Desc = src1MemPtr->getDesc();

        auto src0Shape = src0Desc.getShape();
        auto src0Strides = getStridesAndModifyShape(src0Shape, transposeIn[0]);
        src0TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src0Desc.getPrecision(), src0Shape, src0Strides);

        auto src1Shape = src1Desc.getShape();
        auto src1Strides = getStridesAndModifyShape(src1Shape, transposeIn[1]);
        src1TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src1Desc.getPrecision(), src1Shape, src1Strides);
    } else {
        attr = initPrimitiveAttr();
        src0TransposedDesc = inDataDesc[0];
        src1TransposedDesc = inDataDesc[1];
    }

    auto dstDnnlDesc = dstMemPtr->getDescWithType<DnnlMemoryDesc>();

    DnnlMemoryDescPtr dnnlBiasMemDesc = nullptr;
    if (withBiases) {
        auto biasMemory = getParentEdgeAt(2)->getMemoryPtr();
        if (!biasMemory || !biasMemory->isAllocated())
            IE_THROW()  << errorPrefix << " did not allocate bias memory";
        dnnlBiasMemDesc = biasMemory->getDescWithType<DnnlMemoryDesc>();
    }

    MatMulKey key = {src0TransposedDesc, src1TransposedDesc, dnnlBiasMemDesc,
                     dstDnnlDesc, *attr, selected_pd->getImplementationType()};

    auto engine = getEngine();

    auto builder = [&engine](const MatMulKey& key) -> executorPtr {
        dnnl::matmul::primitive_desc prim_desc;

        if (key.bias) {
            prim_desc = matmul::primitive_desc(
                engine,
                key.inp0->getDnnlDesc(),
                key.inp1->getDnnlDesc(),
                key.bias->getDnnlDesc(),
                key.out->getDnnlDesc(),
                key.attr);
        } else {
            prim_desc = matmul::primitive_desc(
                engine,
                key.inp0->getDnnlDesc(),
                key.inp1->getDnnlDesc(),
                key.out->getDnnlDesc(),
                key.attr);
        }

        auto first_desc = dnnl::matmul::primitive_desc(prim_desc.get());
        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, key.implType);

        if (found)
            return std::make_shared<DnnlExecutor>(prim_desc);

        // In case of dynamic shapes an implementation type chosen as optimal for a primitive_desc with
        // undefined input shapes, is not necessarily available for the primitive_desc with defined shape.
        // Example: brgemm_avx512_amx (Intel Sapphire Rapids Platform) is available for a primitive with
        // undefined input shapes but not available for primitive_desc with input batch 1.
        return std::make_shared<DnnlExecutor>(first_desc);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    execPtr = result.first;
    if (!execPtr) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    auto schratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());

    primArgs[DNNL_ARG_SCRATCHPAD] = schratchpadMem->getPrimitive();
    primArgs[DNNL_ARG_SRC_0] = src0MemPtr->getPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src1MemPtr->getPrimitive();
    primArgs[DNNL_ARG_DST] = dstMemPtr->getPrimitive();
    if (withBiases)
        primArgs[DNNL_ARG_BIAS] = getParentEdgeAt(2)->getMemoryPtr()->getPrimitive();

    appendPostOpArgs(*attr, primArgs, postOpsArgs);
#ifdef CPU_DEBUG_CAPS
    if (result.second == CacheEntryBase::LookUpStatus::Miss) {
        auto pd = execPtr->getPrimitiveDesc();
        DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
    }
#endif
}

void MatMul::execute(dnnl::stream strm) {
    if (execPtr) {
        execPtr->exec(primArgs, strm);
    } else {
        IE_THROW() << errorPrefix << " doesn't have an initialized executor";
    }
}

void MatMul::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

const std::vector<impl_desc_type>& MatMul::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512,
        impl_desc_type::gemm_acl,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_sse42,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm,
        impl_desc_type::jit_gemm,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
        impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,
        impl_desc_type::jit_avx2,
        impl_desc_type::jit_avx_dw,
        impl_desc_type::jit_avx_1x1,
        impl_desc_type::jit_avx,
        impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1,
        impl_desc_type::jit_sse42,
        impl_desc_type::ref,
    };

    return priorities;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
