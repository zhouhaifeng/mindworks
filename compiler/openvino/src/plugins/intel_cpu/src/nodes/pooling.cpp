// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling.h"

#include "fake_quantize.h"
#include "conv.h"
#include "concat.h"
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include <utils/general_utils.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/node_config.h"
#include <common/primitive_hashing_utils.hpp>

// to access and change C pooling primitive desc internal padding field
#include <common/primitive_desc_iface.hpp>
#include <common/pooling_pd.hpp>

#if defined(OV_CPU_WITH_ACL)
#include "executors/acl/acl_utils.hpp"
#include "utils/debug_capabilities.h"
#endif

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct PoolingKey {
    DnnlMemoryDescCPtr inp;
    DnnlMemoryDescCPtr out;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> kernel;
    /// Effective padding. Used to define correct output shape by oneDNN
    /// reshape formula: (iw - kernel + pad_l + pad_r) / strides[i - 2] + 1
    /// should be passed into pooling desc constructor.
    std::vector<ptrdiff_t> effective_pad_begin;
    std::vector<ptrdiff_t> effective_pad_end;
    /// Effective dilation. Used to define correct dilation for OneDNN.
    /// For OneDNN default dilation is vector of zero
    std::vector<ptrdiff_t> effective_dilation;
    std::vector<ptrdiff_t> data_pad_end;
    dnnl::primitive_attr attr;
    dnnl::algorithm alg;
    impl_desc_type implType;

    size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        seed = hash_combine(seed, get_md_hash(*inp->getDnnlDesc().get()));
        seed = get_vector_hash(seed, stride);
        seed = get_vector_hash(seed, kernel);
        seed = get_vector_hash(seed, effective_pad_begin);
        seed = get_vector_hash(seed, effective_pad_end);
        seed = get_vector_hash(seed, effective_dilation);
        seed = get_vector_hash(seed, data_pad_end);
        seed = hash_combine(seed, get_md_hash(*out->getDnnlDesc().get()));
        seed = hash_combine(seed, get_attr_hash(*attr.get()));
        seed = hash_combine(seed, alg);
        seed = hash_combine(seed, implType);
        return seed;
    }

    bool operator==(const PoolingKey& rhs) const {
        bool result = true;
        if (inp != rhs.inp) {
            result = result && inp && rhs.inp && (inp->getDnnlDesc() == rhs.inp->getDnnlDesc());
        }

        if (out != rhs.out) {
            result = result && out && rhs.out && (out->getDnnlDesc() == rhs.out->getDnnlDesc());
        }

        result = result && stride == rhs.stride && kernel == rhs.kernel &&
                 effective_pad_begin == rhs.effective_pad_begin && effective_pad_end == rhs.effective_pad_end &&
                 effective_dilation == rhs.effective_dilation && data_pad_end == rhs.data_pad_end &&
                 *attr.get() == *rhs.attr.get() && alg == rhs.alg && implType == rhs.implType;
        return result;
    }
};

dnnl::pooling_forward::primitive_desc createDescriptorHelper(const dnnl::engine& engine,
                                                             const dnnl::memory::desc& in_candidate,
                                                             const dnnl::memory::desc& out_candidate,
                                                             const dnnl::algorithm alg,
                                                             const std::vector<ptrdiff_t>& stride,
                                                             const std::vector<ptrdiff_t>& kernel,
                                                             const std::vector<ptrdiff_t>& effective_pad_begin,
                                                             const std::vector<ptrdiff_t>& effective_pad_end,
                                                             const std::vector<ptrdiff_t>& effective_dilation,
                                                             const std::vector<ptrdiff_t>& data_pad_end,
                                                             const dnnl::primitive_attr& attr) {
    if (alg == dnnl::algorithm::undef) {
        IE_THROW() << "Unsupported pooling type";
    }

    auto convert = [](std::vector<ptrdiff_t> orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    auto desc = dnnl::pooling_forward::primitive_desc(
        engine,
        prop_kind::forward_inference,
        alg,
        in_candidate,
        out_candidate,
        convert(stride),
        convert(kernel),
        convert(effective_dilation),
        convert(effective_pad_begin),
        convert(effective_pad_end),
        attr,
        true);

    // @ TODO Since oneDNN 3.0 for primitives it is impossible to udpate internal fields of the particular primitive
    // if (alg == dnnl::algorithm::pooling_avg_include_padding) {
        // In case of AVG including paddings the norm coeff should be calculated
        // with tacking into account original pads. So we need to restore
        // original values for end paddings.
        //
        // WA. Because onednn uses different formula to calculate AVG norm coeff
        //     in compare with Caffe. In onednn coeff is always 1/(KH*KW)
        //
        // for (int i = 0; i < data_pad_end.size(); i++) {
        //     if (data_pad_end[i] != effective_pad_end[i]) {
        //         auto pooling_pd = static_cast<dnnl::impl::pooling_pd_t*>(desc_ptr->get());;
        //         pooling_pd->desc()->padding[1][i] = static_cast<ptrdiff_t>(data_pad_end[i]);
        //     }
        // }
    // }

    return desc;
}

}  // namespace

bool Pooling::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (ov::is_type<const ov::op::v8::MaxPool>(op)) {
            if (!op->get_output_target_inputs(1).empty()) {
                errorMessage = "MaxPool from opset8 is supported only with one output";
                return false;
            }
        } else if (!ov::is_type<const ov::op::v1::MaxPool>(op) && !ov::is_type<const ov::op::v1::AvgPool>(op)) {
            errorMessage = "MaxPool and AvgPool from opset1 and MaxPool from opset8 are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Pooling::Pooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto get_attributes = [](std::vector<ptrdiff_t>& internal_attribute, const std::vector<size_t> external_attribute) {
        for (size_t i = 0; i < external_attribute.size(); i++) {
            internal_attribute.push_back(static_cast<ptrdiff_t>(external_attribute[i]));
        }
    };

    if (auto maxPoolOp_v8 = ov::as_type_ptr<const ov::op::v8::MaxPool>(op)) {
        isMaxPool8 = true;
        algorithm = Algorithm::PoolingMax;
        poolingAttrs.exclude_pad = false;
        poolingAttrs.rounding = maxPoolOp_v8->get_rounding_type();
        poolingAttrs.pad_type = maxPoolOp_v8->get_auto_pad();

        get_attributes(poolingAttrs.dilation, maxPoolOp_v8->get_dilations());
        get_attributes(poolingAttrs.stride, maxPoolOp_v8->get_strides());
        get_attributes(poolingAttrs.kernel, maxPoolOp_v8->get_kernel());
        get_attributes(poolingAttrs.data_pad_begin, maxPoolOp_v8->get_pads_begin());
        get_attributes(poolingAttrs.data_pad_end, maxPoolOp_v8->get_pads_end());

        poolingAttrs.auto_pad = (maxPoolOp_v8->get_auto_pad() == ov::op::PadType::SAME_LOWER || maxPoolOp_v8->get_auto_pad() == ov::op::PadType::SAME_UPPER);
    } else if (auto maxPoolOp_v1 = ov::as_type_ptr<const ov::op::v1::MaxPool>(op)) {
        algorithm = Algorithm::PoolingMax;
        poolingAttrs.exclude_pad = false;
        poolingAttrs.pad_type = maxPoolOp_v1->get_auto_pad();
        poolingAttrs.rounding = maxPoolOp_v1->get_rounding_type();

        get_attributes(poolingAttrs.stride, maxPoolOp_v1->get_strides());
        get_attributes(poolingAttrs.kernel, maxPoolOp_v1->get_kernel());
        get_attributes(poolingAttrs.data_pad_begin, maxPoolOp_v1->get_pads_begin());
        get_attributes(poolingAttrs.data_pad_end, maxPoolOp_v1->get_pads_end());
        poolingAttrs.dilation.resize(poolingAttrs.kernel.size(), 1);

        poolingAttrs.auto_pad = (maxPoolOp_v1->get_auto_pad() == ov::op::PadType::SAME_LOWER || maxPoolOp_v1->get_auto_pad() == ov::op::PadType::SAME_UPPER);
    } else if (auto avgPoolOp = ov::as_type_ptr<const ov::op::v1::AvgPool>(op)) {
        algorithm = Algorithm::PoolingAvg;
        poolingAttrs.exclude_pad = avgPoolOp->get_exclude_pad();
        poolingAttrs.rounding = avgPoolOp->get_rounding_type();

        get_attributes(poolingAttrs.stride, avgPoolOp->get_strides());
        get_attributes(poolingAttrs.kernel, avgPoolOp->get_kernel());
        get_attributes(poolingAttrs.data_pad_begin, avgPoolOp->get_pads_begin());
        get_attributes(poolingAttrs.data_pad_end, avgPoolOp->get_pads_end());
        poolingAttrs.dilation.resize(poolingAttrs.kernel.size(), 1);

        poolingAttrs.auto_pad = (avgPoolOp->get_auto_pad() == ov::op::PadType::SAME_LOWER || avgPoolOp->get_auto_pad() == ov::op::PadType::SAME_UPPER);
    }

    poolingAttrs.algorithm = algorithm;
}

std::vector<memory::format_tag> Pooling::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return { memory::format_tag::nCw8c, memory::format_tag::nCw16c, memory::format_tag::nwc, memory::format_tag::ncw};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.getRank() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

void Pooling::initEffectiveAttributes(const Shape &inShape, const Shape &outShape) {
    poolingAttrs.effective_pad_begin = poolingAttrs.data_pad_begin;
    poolingAttrs.effective_pad_end.resize(poolingAttrs.data_pad_end.size());
    poolingAttrs.effective_dilation.resize(poolingAttrs.dilation.size(), 0);

    const auto &inDims = inShape.getStaticDims();
    const auto &outDims = outShape.getStaticDims();

    for (size_t i = 0; i < poolingAttrs.effective_pad_end.size(); i++) {
        int krn = poolingAttrs.kernel[i];
        int dil = poolingAttrs.dilation[i];
        int src = inDims[2 + i];
        int dst = outDims[2 + i];

        int calc_dst = (src - (1 + (krn  - 1) * dil) + poolingAttrs.data_pad_begin[i]) / poolingAttrs.stride[i] + 1;
        poolingAttrs.effective_pad_end[i] = (dst - calc_dst) * poolingAttrs.stride[i];
        poolingAttrs.effective_dilation[i] = dil - 1;
    }
}

void Pooling::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    InferenceEngine::Precision inputPrecision = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outputPrecision = getOriginalOutputPrecisionAtPort(0);

    const auto &parentShape = getInputShapeAtPort(0);
    const auto &childShape = getOutputShapeAtPort(0);
    const size_t inputRank = getInputShapeAtPort(0).getRank();

#if defined(OV_CPU_WITH_ACL)
    // WA: we may specify any layout here (NCHW or NHWC) since both are supported by ACL
    arm_compute::DataLayout dataLayout = (parentShape.getDims().size() == 5) ? arm_compute::DataLayout::NDHWC : arm_compute::DataLayout::NCHW;
    arm_compute::TensorInfo srcTensorInfo = arm_compute::TensorInfo(shapeCast(parentShape.getDims()),
                                                                    1,
                                                                    precisionToAclDataType(inputPrecision),
                                                                    dataLayout);
    arm_compute::TensorInfo dstTensorInfo = arm_compute::TensorInfo(shapeCast(childShape.getDims()),
                                                                    1,
                                                                    precisionToAclDataType(outputPrecision),
                                                                    dataLayout);
    arm_compute::Pooling3dLayerInfo pool3d_info;
    arm_compute::PoolingLayerInfo pool_info;
    useACL = AclPoolingExecutor::isSupported(srcTensorInfo,
                                             dstTensorInfo,
                                             poolingAttrs,
                                             parentShape.getDims().size(),
                                             getOriginalOutputsNumber(),
                                             dataLayout,
                                             (getOriginalOutputsNumber() > 1) ? &getOutputShapeAtPort(1).getDims() : nullptr,
                                             &pool_info,
                                             &pool3d_info);
    //FIXME: 5D tensors case is not assigned to ACL because there is no way to check layout here
    //NEPooling3dLayer supports NDHWC only
    if (parentShape.getDims().size() == 5)
        useACL = false;
#endif
    if (useACL) return;

    // WA: LPT transformation has WA which allows average pooling has I8/U8 output precision instead of FP32,
    // so we explicitly set output precision as FP32
    if (outputPrecision != Precision::I8 && inputPrecision != Precision::BF16) {
        if (getAlgorithm() == Algorithm::PoolingMax) {
            // oneDNN supports only equal precisions for input and output
            outputPrecision = inputPrecision;
        } else if (getAlgorithm() == Algorithm::PoolingAvg) {
            outputPrecision = Precision::FP32;
        }
    }
    if (inputPrecision == Precision::BF16) {
        outputPrecision = inputPrecision;
    }

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    }

    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outputPrecision);

    if ((inputRank < 3) || (inputRank > 5))
        IE_THROW() << "Pooling layer. Unsupported mode. Only 3D, 4D and 5D blobs are supported as input.";

    inShape = MemoryDescUtils::makeDummyShape(parentShape);
    if (isDynamicNode()) {
        const auto& origDims = parentShape.getDims();
        const auto& origMaxDims = parentShape.getMaxDims();

        auto inDims = inShape.getStaticDims();
        for (size_t i = 0; i < inDims.size() - 2; i++) {
            if (origDims[i + 2] == Shape::UNDEFINED_DIM) {
                inDims[i + 2] = std::min<Dim>(origMaxDims[i + 2], std::max<Dim>(inDims[i + 2], poolingAttrs.kernel[i]));
            }
        }
        inShape = Shape(inDims);
    }

    initEffectiveAttributes(inShape,
                            MemoryDescUtils::makeDummyShape(childShape));

    if (inputPrecision == Precision::I8 || inputPrecision == Precision::U8) {
        //  We have to extend i8i8_pooling_fwd_t from oneDNN to support BF16 output data type
        if (outputDataType == memory::data_type::bf16)
            outputDataType = memory::data_type::f32;
        // i8 layers supports only ndhwc and nhwc layouts
        const auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(parentShape, inputDataType, inputRank == 3 ?
                                  memory::format_tag::nwc : (inputRank == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        const auto out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(childShape, outputDataType, inputRank == 3 ?
                                   memory::format_tag::nwc : (inputRank == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        createDescriptor({ in_candidate }, { out_candidate });
    } else if ((inputRank == 3 || inputRank == 4 || inputRank == 5) && parentShape.getDims()[1] == 1) {
        // WA. We should force planar layout since it provides better performance
        const auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(parentShape, inputDataType, inputRank == 3 ?
                                  memory::format_tag::ncw : (inputRank == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw));
        const auto out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(childShape, outputDataType, inputRank == 3 ?
                                   memory::format_tag::ncw : (inputRank == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw));
        createDescriptor({ in_candidate }, { out_candidate });
    } else {
        if (!one_of(inputDataType, memory::data_type::bf16, memory::data_type::f16)) {
            inputDataType = memory::data_type::f32;
            outputDataType = memory::data_type::f32;
        }
        // It doesn't support any format
        for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
            const auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(parentShape, inputDataType, format);
            const auto out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(childShape, outputDataType, format);
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void Pooling::prepareParams() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW()  << "Pooling node with name '" << getName() << "' did not set preferable primitive descriptor";

    AttrPtr attr;
    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = initPrimitiveAttr();
        }
        attr = pAttr;
    } else {
        attr = initPrimitiveAttr();
    }

    if (useACL) {
        auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->isAllocated())
            IE_THROW() << "Destination memory didn't allocate.";
        if (!srcMemPtr || !srcMemPtr->isAllocated())
            IE_THROW() << "Input memory didn't allocate.";

        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
        }

        execPtr = selected_pd->getExecutorFactoryAs<PoolingExecutorFactory>()->makeExecutor(poolingAttrs,
                                                                                            srcMemoryDescs,
                                                                                            dstMemoryDescs,
                                                                                            *attr);
        selected_pd->setImplementationType(execPtr->getImplType());
    } else {
        auto inDesc = getParentEdgesAtPort(0)[0]->getMemory().getDescWithType<DnnlMemoryDesc>();
        auto outDesc = getChildEdgesAtPort(0)[0]->getMemory().getDescWithType<DnnlMemoryDesc>();

        if (isDynamicNode()) {
            if (poolingAttrs.auto_pad) {
                poolingAttrs.data_pad_begin = shapeInference->get_pads_begin();
                poolingAttrs.data_pad_end = shapeInference->get_pads_end();
            }
            initEffectiveAttributes(inDesc->getShape(), outDesc->getShape());
        }

        dnnl::algorithm alg = getPoolingAlgorithm();
        PoolingKey key = {inDesc,
                          outDesc,
                          poolingAttrs.stride,
                          poolingAttrs.kernel,
                          poolingAttrs.effective_pad_begin,
                          poolingAttrs.effective_pad_end,
                          poolingAttrs.effective_dilation,
                          poolingAttrs.data_pad_end,
                          *attr,
                          alg,
                          selected_pd->getImplementationType()};
        auto engine = getEngine();
        auto builder = [&engine](const PoolingKey& key) -> executorPtr {
            auto prim_desc = createDescriptorHelper(
                engine,
                key.inp->getDnnlDesc(),
                key.out->getDnnlDesc(),
                key.alg,
                key.stride,
                key.kernel,
                key.effective_pad_begin,
                key.effective_pad_end,
                key.effective_dilation,
                key.data_pad_end,
                key.attr);

            auto first_desc = dnnl::pooling_forward::primitive_desc(prim_desc.get());
            const bool found = DnnlExtensionUtils::find_implementation(prim_desc, key.implType);

            if (found)
                return std::make_shared<DnnlExecutor>(prim_desc);

            // use the first available
            return std::make_shared<DnnlExecutor>(first_desc);
        };

        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);

        dnnlExecPtr = result.first;

        if (!dnnlExecPtr) {
            IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
        }

        auto scratchpadMem = getScratchPadMem(dnnlExecPtr->getScratchPadDesc());
        primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();
        primArgs[DNNL_ARG_SRC] = getParentEdgesAtPort(0)[0]->getMemoryPtr()->getPrimitive();
        primArgs[DNNL_ARG_DST] = getChildEdgesAtPort(0)[0]->getMemoryPtr()->getPrimitive();

        Node::appendPostOpArgs(*attr, primArgs, postOpsArgs);

#ifdef CPU_DEBUG_CAPS
        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
            auto pd = dnnlExecPtr->getPrimitiveDesc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
    }
}

void Pooling::execute(dnnl::stream strm) {
    if (dnnlExecPtr) {
        dnnlExecPtr->exec(primArgs, strm);
    } else if (execPtr) {
        std::vector<MemoryCPtr> srcMemory;
        for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
        }
        std::vector<MemoryPtr> dstMemory;
        for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
        }

        execPtr->exec(srcMemory, dstMemory, postOpsArgs);
    } else {
        IE_THROW() << "Pooling node with name '" << getName() << "' doesn't have an initialized executor";
    }
}

void Pooling::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Pooling::created() const {
    return getType() == Type::Pooling;
}

dnnl::algorithm Pooling::getPoolingAlgorithm() const {
    if (algorithm == Algorithm::PoolingAvg) {
        bool not_zero_l = false;
        for (auto lr : poolingAttrs.data_pad_begin) {
            if (lr) {
                not_zero_l = true;
                break;
            }
        }
        bool not_zero_r = false;
        for (auto pr : poolingAttrs.data_pad_end) {
            if (pr) {
                not_zero_r = true;
                break;
            }
        }
        if (!poolingAttrs.exclude_pad && (not_zero_l || not_zero_r))
            return dnnl::algorithm::pooling_avg_include_padding;
        else
            return dnnl::algorithm::pooling_avg_exclude_padding;
    } else if (algorithm == Algorithm::PoolingMax) {
        return dnnl::algorithm::pooling_max;
    } else {
        return dnnl::algorithm::undef;
    }
}

dnnl::pooling_forward::primitive_desc Pooling::createDescriptorInternal(
    const dnnl::memory::desc& in_candidate,
    const dnnl::memory::desc& out_candidate,
    const dnnl::algorithm alg) {

    auto attr = initPrimitiveAttr();

    return createDescriptorHelper(getEngine(),
                                  in_candidate,
                                  out_candidate,
                                  alg,
                                  poolingAttrs.stride,
                                  poolingAttrs.kernel,
                                  poolingAttrs.effective_pad_begin,
                                  poolingAttrs.effective_pad_end,
                                  poolingAttrs.effective_dilation,
                                  poolingAttrs.data_pad_end,
                                  *attr);
}

void Pooling::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                               const std::vector<MemoryDescPtr> &outputDesc) {
    auto inDesc = inputDesc[0]->isDefined() ? inputDesc[0] : inputDesc[0]->cloneWithNewDims(inShape.getStaticDims());
    auto dnnlInDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inDesc);
    auto in_candidate = dnnlInDesc->getDnnlDesc();

    auto outDesc = outputDesc[0];
    if (!outDesc->isDefined()) {
        auto outDims = shapeInferGeneric({Shape(inDesc->getShape().getStaticDims())});
        outDesc = outDesc->cloneWithNewDims(outDims[0]);
        if (poolingAttrs.auto_pad) {
            poolingAttrs.data_pad_begin = shapeInference->get_pads_begin();
            poolingAttrs.data_pad_end = shapeInference->get_pads_end();
        }
        initEffectiveAttributes(inDesc->getShape(), outDesc->getShape());
    }
    auto dnnlOutDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*outDesc);
    const auto& out_candidate = dnnlOutDesc.getDnnlDesc();

    auto desc = createDescriptorInternal(in_candidate, out_candidate, getPoolingAlgorithm());

    if (desc)
        descs.emplace_back(desc);
}

void Pooling::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (useACL) {
        auto& creatorsMap = BlockedDescCreator::getCommonCreators();
        auto pushDesc = [&](LayoutType format) {
            NodeConfig config;
            config.inConfs.resize(getParentEdges().size());
            config.outConfs.resize(getOriginalOutputsNumber());

            config.inConfs[0].setMemDesc(
                creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(0)));
            config.outConfs[0].setMemDesc(
                creatorsMap.at(format)->createSharedDesc(getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0)));

            std::vector<MemoryDescPtr> srcMemoryDescs;
            for (const auto& inConf : config.inConfs) {
                srcMemoryDescs.push_back(inConf.getMemDesc());
            }
            std::vector<MemoryDescPtr> dstMemoryDescs;
            for (const auto& outConf : config.outConfs) {
                dstMemoryDescs.push_back(outConf.getMemDesc());
            }

            auto factory = std::make_shared<PoolingExecutorFactory>(
                poolingAttrs,
                srcMemoryDescs,
                dstMemoryDescs,
                std::make_shared<ExecutorContext>(context, getImplPriority()));
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef, factory);
        };

        pushDesc(LayoutType::ncsp);

        return;
    }

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = canBeInPlace() ? 0 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);
            inConfs.emplace_back(desc);
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);
            // PortConfig in{desc, inPlaceOutPort};
            outConfs.emplace_back(desc, BlockedMemoryDesc::FULL_MASK, inPlaceOutPort);
        }

        // CPU plugin doesn't support second output of MaxPool-8, but anyway we should have out config for second port as stub
        if (isMaxPool8) {
            const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
            const auto outputPrecision = outConfs.front().getMemDesc()->getPrecision();
            auto desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outputPrecision, getOutputShapeAtPort(1));

            outConfs.emplace_back(desc);
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

void Pooling::initDescriptor(const NodeConfig& config) {
    if (useACL)
        return;

    Node::initDescriptor(config);
}

Node::AttrPtr Pooling::initPrimitiveAttr() {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr);

    (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    return attr;
}

void Pooling::setPostOps(dnnl::primitive_attr &attr) {
    dnnl::post_ops ops;

    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsArgs);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

}  // namespace node
}   // namespace intel_cpu
}   // namespace ov
