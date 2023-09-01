/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "DotOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

triton::nvgpu::WGMMAEltType getMmaRetType(Value d) {
  auto dTy = d.getType().cast<RankedTensorType>().getElementType();
  if (dTy.isF32()) {
    return triton::nvgpu::WGMMAEltType::f32;
  } else if (dTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (dTy.isInteger(32)) {
    return triton::nvgpu::WGMMAEltType::s32;
  } else {
    llvm::report_fatal_error("Unsupported mma result type found");
  }
}

triton::nvgpu::WGMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = a.getType().cast<RankedTensorType>().getElementType();
  if (aTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (aTy.isBF16()) {
    return triton::nvgpu::WGMMAEltType::bf16;
  } else if (aTy.isF32() && allowTF32) {
    return triton::nvgpu::WGMMAEltType::tf32;
  } else if (aTy.isInteger(8)) {
    return triton::nvgpu::WGMMAEltType::s8;
  } else if (aTy.isFloat8E5M2()) {
    return triton::nvgpu::WGMMAEltType::e5m2;
  } else if (aTy.isFloat8E4M3FNUZ()) {
    return triton::nvgpu::WGMMAEltType::e4m3;
  } else {
    llvm::report_fatal_error("Unsupported mma operand type found");
  }
}

mlir::triton::nvgpu::WGMMADescMode
getModeFromLayout(const SharedEncodingAttr &layout, uint32_t widthInByte) {
  int perPhase = layout.getPerPhase();
  int maxPhase = layout.getMaxPhase();
  uint32_t swizzlingByteWidth = 0;

  mlir::triton::nvgpu::WGMMADescMode mode;
  if (perPhase == 4 && maxPhase == 2) {
    mode = mlir::triton::nvgpu::WGMMADescMode::swizzle32;
    swizzlingByteWidth = 32;
  } else if (perPhase == 2 && maxPhase == 4) {
    mode = mlir::triton::nvgpu::WGMMADescMode::swizzle64;
    swizzlingByteWidth = 64;
  } else if (perPhase == 1 && maxPhase == 8) {
    mode = mlir::triton::nvgpu::WGMMADescMode::swizzle128;
    swizzlingByteWidth = 128;
  } else {
    llvm::report_fatal_error("Unsupported shared layout.");
  }

  // TODO[biaow]: remove it once we support swizzling size larger than matrix
  // width, which requires padding the matrix width to the swizzling size when
  // allocating shared memory.
  assert(swizzlingByteWidth <= widthInByte &&
         "swizzling size larger than matrix width is not supported.");
  return mode;
}

class DotOpMmaV3SmemLoader {
public:
  DotOpMmaV3SmemLoader(Value tensor, const SharedMemoryObject &smemObj,
                       SmallVector<int64_t> shape, Value warpId,
                       unsigned int dimWpt, bool trans,
                       SmallVector<unsigned int> instrShape,
                       ConversionPatternRewriter &rewriter, Location loc)
      : base(smemObj.base), shape(shape), warpId(warpId), dimWpt(dimWpt),
        trans(trans), instrShape(instrShape), rewriter(rewriter), loc(loc) {
    auto tensorTy = tensor.getType().cast<RankedTensorType>();
    auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
    ord = sharedLayout.getOrder();
    const int perPhase = sharedLayout.getPerPhase();
    const int maxPhase = sharedLayout.getMaxPhase();
    elemBytes = tensorTy.getElementTypeBitWidth() / 8;
    elemsPerSwizzlingRow = 128 / perPhase / elemBytes;
    elemsPerSwizzlingRowVal = i32_val(elemsPerSwizzlingRow);

    uint32_t widthInByte = shape[ord[0]] * elemBytes;
    mode = getModeFromLayout(sharedLayout, widthInByte);

    baseDesc = rewriter.create<triton::nvgpu::WGMMADescCreateOp>(
        loc, i64_ty, base, i32_val(shape[ord[1]]), mode);
  }

  Value smemLoad(int a, int b) {
    Value k = i32_val(b * instrShape[1]);
    Value m = add(i32_val(a * dimWpt * instrShape[0]),
                  mul(warpId, i32_val(instrShape[0])));
    if (trans) {
      std::swap(k, m);
    }
    Value leading_offset = mul(udiv(k, elemsPerSwizzlingRowVal),
                               i32_val(shape[ord[1]] * elemsPerSwizzlingRow));
    Value stride_offset = mul(m, elemsPerSwizzlingRowVal);
    Value offset = add(add(leading_offset, stride_offset),
                       urem(k, elemsPerSwizzlingRowVal));
    Value off1 = mul(i32_val(elemBytes), offset);
    Value off_ = zext(i64_ty, udiv(off1, i32_val(16)));

    return add(baseDesc, off_);
  }

private:
  Value base;
  SmallVector<int64_t> shape;
  Value warpId;
  int dimWpt;
  bool trans;
  Value elemsPerSwizzlingRowVal;
  mlir::triton::nvgpu::WGMMADescMode mode;
  SmallVector<unsigned int> instrShape;
  ArrayRef<unsigned> ord;
  ConversionPatternRewriter &rewriter;
  Location loc;
  int elemsPerSwizzlingRow;
  int elemBytes;
  Value baseDesc;
};

DotOpMmaV3SmemLoader loadA(TritonGPUToLLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const MmaEncodingAttr &mmaEncoding, Value tensor,
                           const SharedMemoryObject &smemObj, Value thread) {
  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  auto aSharedLayout = aTensorTy.getEncoding().dyn_cast<SharedEncodingAttr>();
  assert(aSharedLayout && "only support load dot operand from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  auto aOrd = aSharedLayout.getOrder();
  bool transA = aOrd[0] == 0;
  auto shapePerCTA = getShapePerCTA(aTensorTy);

  int numRepM = ceil<unsigned>(shapePerCTA[0], instrShape[0] * wpt[0]);
  int numRepK = ceil<unsigned>(shapePerCTA[1], instrShape[2]);

  Value warp = udiv(thread, i32_val(32));
  Value warpM = urem(warp, i32_val(wpt[0]));
  Value warpId = urem(warpM, i32_val(shapePerCTA[0] / instrShape[0]));

  return {tensor,
          smemObj,
          shapePerCTA,
          warpId,
          wpt[0],
          transA,
          {instrShape[0], instrShape[2]},
          rewriter,
          loc};
}

DotOpMmaV3SmemLoader loadB(TritonGPUToLLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           MmaEncodingAttr &mmaEncoding, Value tensor,
                           const SharedMemoryObject &smemObj, Value thread) {
  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  auto bSharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  assert(bSharedLayout && "only support load B from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  auto bOrd = bSharedLayout.getOrder();
  bool transB = bOrd[0] == 1;
  auto shapePerCTA = triton::gpu::getShapePerCTA(bTensorTy);

  int numRepK = ceil<unsigned>(shapePerCTA[0], instrShape[2]);
  int numRepN = ceil<unsigned>(shapePerCTA[1], instrShape[1] * wpt[1]);

  Value warp = udiv(thread, i32_val(32));
  Value warpMN = udiv(warp, i32_val(wpt[0]));
  Value warpN = urem(warpMN, i32_val(wpt[1]));
  Value warpId = urem(warpN, i32_val(shapePerCTA[1] / instrShape[1]));

  return {tensor,
          smemObj,
          shapePerCTA,
          warpId,
          wpt[1],
          transB,
          {instrShape[1], instrShape[2]},
          rewriter,
          loc};
}

// Return a vector of Value of the accumulator start at startIndex and pack the
// values into 32bits in case the accumulator is fp16.
llvm::SmallVector<Value> loadC(ConversionPatternRewriter &rewriter,
                               Location loc, const SmallVector<Value> &elements,
                               int startIndex, int numElements) {
  if (!elements[0].getType().isF16()) {
    llvm::SmallVector<Value> mmaOut(numElements);
    for (int i = 0; i < numElements; ++i)
      mmaOut[i] = elements[startIndex + i];
    return mmaOut;
  }
  // For FP16 we need to pack accumulator into 32-bit integers.
  llvm::SmallVector<Value> mmaOut(numElements / 2);
  for (int i = 0; i < numElements / 2; ++i) {
    Value a0 = elements[startIndex + 2 * i];
    Value a1 = elements[startIndex + 2 * i + 1];
    Type cPackTy = vec_ty(rewriter.getF16Type(), 2);
    Value pack = rewriter.create<LLVM::UndefOp>(loc, cPackTy);
    pack = insert_element(cPackTy, pack, a0, i32_val(0));
    pack = insert_element(cPackTy, pack, a1, i32_val(1));
    pack = bitcast(pack, rewriter.getIntegerType(32));
    mmaOut[i] = pack;
  }
  return mmaOut;
}

// If the accumulator is fp16 unpack it from 32-bit integers.
SmallVector<Value> unpackAccumulator(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const SmallVector<Value> &packed,
                                     RankedTensorType tensorTy) {
  if (!tensorTy.getElementType().isF16())
    return packed;
  // For fp16 the accumualtor is pack into 32-bit integers so we need to unpack
  // it.
  SmallVector<Value> results;
  for (Value elem : packed) {
    elem = bitcast(elem, vec_ty(rewriter.getF16Type(), 2));
    results.push_back(extract_element(rewriter.getF16Type(), elem, i32_val(0)));
    results.push_back(extract_element(rewriter.getF16Type(), elem, i32_val(1)));
  }
  return results;
}

LogicalResult convertDot(TritonGPUToLLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value loadedA, Value loadedB, Value loadedC,
                         bool allowTF32, const SharedMemoryObject &smemObjA,
                         const SharedMemoryObject &smemObjB, bool sync,
                         Value thread) {
  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto bTensorTy = b.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();
  auto aSharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto bSharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto mmaEncoding = dTensorTy.getEncoding().cast<MmaEncodingAttr>();
  auto aOrd = aSharedLayout.getOrder();
  auto bOrd = bSharedLayout.getOrder();
  bool transA = aOrd[0] == 0;
  bool transB = bOrd[0] == 1;
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrShape = mmaEncoding.getInstrShape();
  auto accSize = 2 * (instrShape[1] / 4);
  int M = 4 * instrShape[0];
  int N = instrShape[1];
  int K = instrShape[2];

  auto shapePerCTATile = getShapePerCTATile(mmaEncoding);
  int numRepM = ceil<unsigned>(dShapePerCTA[0], shapePerCTATile[0]);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], shapePerCTATile[1]);
  int numRepK = ceil<unsigned>(aTensorTy.getShape()[1], instrShape[2]);

  DotOpMmaV3SmemLoader aLoader =
      loadA(typeConverter, rewriter, loc, mmaEncoding, a, smemObjA, thread);
  DotOpMmaV3SmemLoader bLoader =
      loadB(typeConverter, rewriter, loc, mmaEncoding, b, smemObjB, thread);

  auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter, dTensorTy);

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  int numTMADescs =
      func->getAttr(kAttrNumTMALoadDescsName).cast<IntegerAttr>().getInt();
  if (numTMADescs == 0)
    rewriter.create<triton::nvgpu::FenceAsyncSharedOp>(loc, 0);
  rewriter.create<triton::nvgpu::WGMMAFenceOp>(loc);

  SmallVector<Value> mmaResults;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      llvm::SmallVector<Value> mmaOut =
          loadC(rewriter, loc, fc, (m * numRepN + n) * accSize, accSize);
      llvm::SmallVector<Type> elemTypes;
      for (Value accEl : mmaOut)
        elemTypes.push_back(accEl.getType());
      auto accTy =
          LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);
      Value d = typeConverter->packLLElements(loc, mmaOut, rewriter, accTy);
      for (int k = 0; k < numRepK; ++k) {
        auto a = aLoader.smemLoad(m, k);
        auto b = bLoader.smemLoad(n, k);
        ValueRange operands{a, b, d};
        d = rewriter.create<triton::nvgpu::WGMMAOp>(loc, accTy, a, b, d, M, N,
                                                    K, eltTypeC, eltTypeA,
                                                    eltTypeB, layoutA, layoutB);
      }
      auto acc = typeConverter->unpackLLElements(loc, d, rewriter, accTy);
      for (int i = 0; i < acc.size(); ++i) {
        mmaResults.push_back(acc[i]);
      }
    }
  }
  rewriter.create<triton::nvgpu::WGMMACommitGroupOp>(loc);

  if (sync)
    rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(loc, 0);

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = typeConverter->packLLElements(loc, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

// Loading $c to registers, returns a Value.
Value loadC(Value tensor, Value llTensor) {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  auto mmaEncoding = tensorTy.getEncoding().dyn_cast<MmaEncodingAttr>();
  assert(mmaEncoding && "Currently, we only support $c with a mma layout.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  auto shapePerCTA = getShapePerCTA(tensorTy);
  auto shapePerCTATile = getShapePerCTATile(mmaEncoding);

  int numRepM = ceil<unsigned>(shapePerCTA[0], shapePerCTATile[0]);
  int numRepN = ceil<unsigned>(shapePerCTA[1], shapePerCTATile[1]);

  size_t fcSize = 2 * (instrShape[1] / 4) * numRepM * numRepN;

  auto structTy = llTensor.getType().cast<LLVM::LLVMStructType>();
  assert(structTy.getBody().size() == fcSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");
  return llTensor;
}

LogicalResult convertWGMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                           TritonGPUToLLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread) {
  auto loc = op.getLoc();
  Value A = op.getA();
  Value B = op.getB();
  Value C = op.getC();
  auto ATensorTy = A.getType().cast<RankedTensorType>();
  auto BTensorTy = B.getType().cast<RankedTensorType>();

  assert(ATensorTy.getEncoding().isa<SharedEncodingAttr>() &&
         BTensorTy.getEncoding().isa<SharedEncodingAttr>() &&
         "Both $a and %b should be Shared layout.");

  Value llA, llB, llC;
  llA = adaptor.getA();
  llB = adaptor.getB();
  llC = loadC(C, adaptor.getC());

  auto smemObjA = getSharedMemoryObjectFromStruct(loc, llA, rewriter);
  auto smemObjB = getSharedMemoryObjectFromStruct(loc, llB, rewriter);
  return convertDot(typeConverter, rewriter, loc, op.getOperation(), A, B, C,
                    op.getD(), llA, llB, llC, op.getAllowTF32(), smemObjA,
                    smemObjB, true, thread);
}

LogicalResult convertAsyncWGMMA(triton::nvidia_gpu::DotAsyncOp op,
                                triton::nvidia_gpu::DotAsyncOp::Adaptor adaptor,
                                TritonGPUToLLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter,
                                Value thread) {
  auto loc = op.getLoc();
  Value A = op.getA();
  Value B = op.getB();
  Value C = op.getC();
  auto ATensorTy = A.getType().cast<RankedTensorType>();
  auto BTensorTy = B.getType().cast<RankedTensorType>();

  assert(ATensorTy.getEncoding().isa<SharedEncodingAttr>() &&
         BTensorTy.getEncoding().isa<SharedEncodingAttr>() &&
         "Both $a and %b should be Shared layout.");

  Value llA, llB, llC;
  llA = adaptor.getA();
  llB = adaptor.getB();
  llC = loadC(C, adaptor.getC());

  auto smemObjA = getSharedMemoryObjectFromStruct(loc, llA, rewriter);
  auto smemObjB = getSharedMemoryObjectFromStruct(loc, llB, rewriter);
  return convertDot(typeConverter, rewriter, loc, op.getOperation(), A, B, C,
                    op.getD(), llA, llB, llC, op.getAllowTF32(), smemObjA,
                    smemObjB, false, thread);
}
