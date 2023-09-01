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

#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <memory>
#include <stack>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {
bool isDivisible(Value v, unsigned divisor) {
  if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    return op.getValue().dyn_cast<IntegerAttr>().getValue().getZExtValue() %
               divisor ==
           0;
  }
  if (v.getDefiningOp() &&
      isa<mlir::UnrealizedConversionCastOp>(v.getDefiningOp())) {
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);
  } else if (v.getParentBlock()->isEntryBlock() && v.isa<BlockArgument>()) {
    BlockArgument blockArg = v.cast<BlockArgument>();
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    auto func = dyn_cast<tt::FuncOp>(parentOp);
    assert(func);
    if (auto attr = func.getArgAttrOfType<IntegerAttr>(blockArg.getArgNumber(),
                                                       "tt.max_divisibility"))
      return attr.getValue().getZExtValue() % divisor == 0;
    return false;
  } else if (v.getParentBlock()->isEntryBlock() && (!v.isa<BlockArgument>())) {
    // in entryblock but not BlockArgument
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);
  } else if (!v.getParentBlock()->isEntryBlock()) {
    // in non-entryblock
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);
  } else {
    llvm::report_fatal_error(
        "Operand of `MakeTensorPtrOp` is not the function's argument");
    return false;
  }
}

bool shouldRemove(tt::MakeTensorPtrOp &op, int computeCapability) {
  if (computeCapability < 90 || !::triton::tools::getBoolEnv("ENABLE_TMA"))
    return true;
  auto resType = op.getResult()
                     .getType()
                     .cast<tt::PointerType>()
                     .getPointeeType()
                     .cast<RankedTensorType>();
  auto elemType = resType.getElementType();
  auto ord = op.getOrder();
  auto stride = op.getStrides();
  auto shape = ttg::getShapePerCTA(resType);
  // TMA load/store requires the box dimension to be more than 32 bytes.
  // Because we only support 32B-swizzle, 64B-swizzle and 128B-swizzleon for
  // now. Remove this constraint when we support non-swizzle smem.
  bool boxDimSwizzle =
      shape[ord[0]] >= (256 / elemType.getIntOrFloatBitWidth());
  // we only support TMA load with 2D tensor for now.
  // TMA load/store requires the stride to be divisible by 16 bytes.
  bool strideDivisible = false;
  if (stride.size() == 2)
    strideDivisible =
        isDivisible(stride[ord[1]], 128 / elemType.getIntOrFloatBitWidth());
  bool enableTMA = ::triton::tools::getBoolEnv("ENABLE_TMA");
  return !(boxDimSwizzle && strideDivisible && enableTMA);
}

// TODO: When encoding exists use triton::gpu::CmpIOp as arith::CmpIOp doesn't
// play well with encoding attributes. Move back to arith::CmpIOp when this pass
// moves back to triton IR level.
Value createCmpOp(OpBuilder &builder, Location loc, RankedTensorType type,
                  arith::CmpIPredicate pred, Value lhs, Value rhs) {
  if (type.getEncoding())
    return builder.create<ttg::CmpIOp>(loc, type, pred, lhs, rhs);
  return builder.create<arith::CmpIOp>(loc, type, pred, lhs, rhs);
}

/// An additional struct to record the meta information of operations
/// with tensor pointers
struct RewritedInfo {
private:
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  ArrayRef<int64_t> tensorShape;
  Attribute layout;

  // A cache to avoid generating the same offset with range
  DenseMap<unsigned, Value> cachedOffsetWithRange;

  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }

public:
  RewritedInfo() = default;

  RewritedInfo(const RewritedInfo &other) = default;

  RewritedInfo(Value base, const SmallVector<Value> &shape,
               const SmallVector<Value> &strides,
               const SmallVector<Value> &offsets,
               const ArrayRef<int64_t> &tensorShape, Attribute layout)
      : base(base), shape(shape), strides(strides), offsets(offsets),
        tensorShape(tensorShape), layout(layout) {
    assert(shape.size() == strides.size() && shape.size() == offsets.size() &&
           shape.size() == tensorShape.size());
  }

  unsigned int length() const { return shape.size(); }

  Value getOffset(unsigned i) { return offsets[i]; }

  SmallVector<Value> getOffsets() { return offsets; }

  void setOffset(unsigned i, Value newOffset) {
    offsets[i] = newOffset;
    cachedOffsetWithRange.clear();
  }

  void setOffsets(const SmallVector<Value> &newOffsets) {
    offsets = newOffsets;
    cachedOffsetWithRange.clear();
  }

  void setEncoding(Attribute newLayout) { layout = newLayout; }

  Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                   unsigned i) {
    if (cachedOffsetWithRange.count(i))
      return cachedOffsetWithRange[i];

    // Add range
    auto indexI32RowType =
        RankedTensorType::get({tensorShape[i]}, builder.getI32Type(), layout);
    auto indexRowType =
        RankedTensorType::get({tensorShape[i]}, builder.getI64Type(), layout);
    Value splatOffset =
        builder.create<tt::SplatOp>(loc, indexRowType, offsets[i]);
    Value range = builder.create<tt::MakeRangeOp>(loc, indexI32RowType, 0,
                                                  tensorShape[i]);
    Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

    // Expand dimensions
    Value expandedResult =
        builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
    for (int axis = 0; axis < tensorShape.size(); ++axis) {
      if (axis == i)
        continue;

      if (layout) {
        auto argEncoding = layout.cast<ttg::BlockedEncodingAttr>();
        auto retSizePerThread = insertOne(argEncoding.getSizePerThread(), axis);
        auto retThreadsPerWarp =
            insertOne(argEncoding.getThreadsPerWarp(), axis);
        auto retWarpsPerCTA = insertOne(argEncoding.getWarpsPerCTA(), axis);
        auto retOrder = insertOrder(argEncoding.getOrder(), axis);

        auto argCTALayout = argEncoding.getCTALayout();
        auto retCTAsPerCGA = insertOne(argCTALayout.getCTAsPerCGA(), axis);
        auto retCTASplitNum = insertOne(argCTALayout.getCTASplitNum(), axis);
        auto retCTAOrder = insertOrder(argCTALayout.getCTAOrder(), axis);

        auto retCTALayout = ttg::CTALayoutAttr::get(
            loc.getContext(), retCTAsPerCGA, retCTASplitNum, retCTAOrder);

        auto retEncoding = ttg::BlockedEncodingAttr::get(
            loc.getContext(), retSizePerThread, retThreadsPerWarp,
            retWarpsPerCTA, retOrder, retCTALayout);

        auto newArgEncoding =
            ttg::SliceEncodingAttr::get(loc.getContext(), axis, retEncoding);
        auto newArgType = RankedTensorType::get(indexRowType.getShape(),
                                                indexRowType.getElementType(),
                                                newArgEncoding);
        Value newArg = builder.create<ttg::ConvertLayoutOp>(loc, newArgType,
                                                            expandedResult);
        expandedResult = builder.create<tt::ExpandDimsOp>(loc, newArg, axis);
      } else
        expandedResult =
            builder.create<tt::ExpandDimsOp>(loc, expandedResult, axis);
    }

    return cachedOffsetWithRange[i] = expandedResult;
  }

  Value generatePtr(OpBuilder &builder, const Location &loc) {
    assert(tensorShape.size() == offsets.size() &&
           tensorShape.size() == strides.size());
    auto ptrType = base.getType().cast<tt::PointerType>();
    auto ptrTensorType = RankedTensorType::get(tensorShape, ptrType, layout);

    // Generate offsets per dimension
    Value ptr = builder.create<tt::SplatOp>(loc, ptrTensorType, base);
    for (unsigned i = 0; i < tensorShape.size(); ++i) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);
      // We must splat strides into the expanded shape not a row for retaining
      // the divisibility information given by strides
      Value splatStride = builder.create<tt::SplatOp>(
          loc, offsetWithRange.getType(), strides[i]);
      Value offsetWithStride =
          builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
      auto offsetType = offsetWithRange.getType().cast<RankedTensorType>();
      auto indexTensorType = RankedTensorType::get(
          tensorShape, offsetType.getElementType(), offsetType.getEncoding());
      Value broadcasted = builder.create<tt::BroadcastOp>(loc, indexTensorType,
                                                          offsetWithStride);
      if (offsetType.getEncoding() != ptrTensorType.getEncoding()) {
        auto newArgType =
            RankedTensorType::get(tensorShape, offsetType.getElementType(),
                                  ptrTensorType.getEncoding());
        broadcasted =
            builder.create<ttg::ConvertLayoutOp>(loc, newArgType, broadcasted);
      }
      // Add to the pointer
      ptr = builder.create<tt::AddPtrOp>(loc, ptrTensorType, ptr, broadcasted);
    }

    return ptr;
  }

  Value generateMask(OpBuilder &builder, const Location &loc,
                     const std::optional<ArrayRef<int32_t>> &boundaryCheck) {
    if (!boundaryCheck.has_value() || boundaryCheck.value().empty())
      return {};

    // Generate mask per dimension
    auto maskTensorType =
        RankedTensorType::get(tensorShape, builder.getI1Type(), layout);
    Value mask;
    for (auto i : boundaryCheck.value()) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);
      auto offsetType = offsetWithRange.getType().cast<RankedTensorType>();
      RankedTensorType cmpTensorType = RankedTensorType::get(
          offsetType.getShape(), builder.getI1Type(), offsetType.getEncoding());

      // Compare with lower bound
      Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
          loc, 0, offsetType.getElementType());
      Value splatLowerBound = builder.create<tt::SplatOp>(
          loc, offsetWithRange.getType(), lowerBound);
      Value cmpLower =
          createCmpOp(builder, loc, cmpTensorType, arith::CmpIPredicate::sge,
                      offsetWithRange, splatLowerBound);

      // Compare with upper bound
      Value splatUpperBound =
          builder.create<tt::SplatOp>(loc, offsetWithRange.getType(), shape[i]);
      Value cmpUpper =
          createCmpOp(builder, loc, cmpTensorType, arith::CmpIPredicate::slt,
                      offsetWithRange, splatUpperBound);

      // And and broadcast
      Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
      if (offsetType.getEncoding() != maskTensorType.getEncoding()) {
        auto newArgType =
            RankedTensorType::get(offsetType.getShape(), builder.getI1Type(),
                                  maskTensorType.getEncoding());
        andResult =
            builder.create<ttg::ConvertLayoutOp>(loc, newArgType, andResult);
      }

      Value broadcasted =
          builder.create<tt::BroadcastOp>(loc, maskTensorType, andResult);

      // And up all results
      if (!mask) {
        mask = broadcasted;
      } else {
        mask = builder.create<arith::AndIOp>(loc, mask, broadcasted);
      }
    }

    return mask;
  }

  Value generateOther(OpBuilder &builder, const Location &loc,
                      const std::optional<tt::PaddingOption> &padding) {
    if (!padding.has_value())
      return Value();

    // Create element attribute
    auto elementType = base.getType().cast<tt::PointerType>().getPointeeType();
    auto otherTensorType =
        RankedTensorType::get(tensorShape, elementType, layout);

    // Set zero padding value
    TypedAttr attr =
        elementType.isIntOrIndex()
            ? builder.getIntegerAttr(elementType, 0).cast<TypedAttr>()
            : builder.getFloatAttr(elementType, 0).cast<TypedAttr>();

    // Float NaN padding case
    if (padding.value() == tt::PaddingOption::PAD_NAN) {
      assert(!elementType.isIntOrIndex());
      auto apNaN = llvm::APFloat::getNaN(
          attr.cast<FloatAttr>().getValue().getSemantics());
      attr = builder.getFloatAttr(elementType, apNaN);
    }

    // Create tensor
    Value constant = builder.create<arith::ConstantOp>(loc, attr);
    return builder.create<tt::SplatOp>(loc, otherTensorType, constant);
  }
};
} // namespace

class TritonGPURewriteTensorPointerPass
    : public TritonGPURewriteTensorPointerBase<
          TritonGPURewriteTensorPointerPass> {
private:
  int computeCapability;
  DenseMap<Value, RewritedInfo> rewritedInfo;

public:
  explicit TritonGPURewriteTensorPointerPass(int computeCapability)
      : computeCapability(computeCapability) {}

  static bool needRewrite(Operation *op, const DenseSet<Value> &valueToRemove) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (op->getNumResults() == 0)
        return false;
      Operation *thenYield = ifOp.thenYield().getOperation();
      if (!ifOp.getElseRegion().empty()) {
        Operation *elseYield = ifOp.elseYield().getOperation();
        for (unsigned i = 0; i < thenYield->getNumOperands(); ++i) {
          bool thenNeedRewrite = valueToRemove.count(thenYield->getOperand(i));
          bool elseNeedRewrite = valueToRemove.count(elseYield->getOperand(i));
          assert(!(thenNeedRewrite ^ elseNeedRewrite) &&
                 "For IfOp, operand(i) of thenYield and operand(i) of "
                 "elseYield should be either all need rewrite or all not");
        }
      }
      op = thenYield;
    }
    return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                       [&valueToRemove](Value operand) {
                         return tt::isTensorPointerType(operand.getType()) &&
                                valueToRemove.count(operand);
                       });
  }

  static SmallVector<Value>
  generateNewOperands(const SmallVector<Value> &oldOperands, unsigned index,
                      const SmallVector<Value> &newValues) {
    assert(index < oldOperands.size());
    SmallVector<Value> newOperands;
    for (int i = 0; i < index; ++i)
      newOperands.push_back(oldOperands[i]);
    for (auto value : newValues)
      newOperands.push_back(value);
    for (auto i = index + 1; i < oldOperands.size(); ++i)
      newOperands.push_back(oldOperands[i]);
    return newOperands;
  }

  Operation *rewriteMakeTensorPtrOp(OpBuilder &builder, tt::MakeTensorPtrOp op,
                                    std::stack<Operation *> &eraser,
                                    const DenseSet<Value> &valueToRemove) {
    if (!valueToRemove.count(op.getResult()))
      return nullptr;
    // Save info for later use
    auto ptrType = op.getResult().getType().cast<tt::PointerType>();
    auto tensorType = ptrType.getPointeeType().cast<RankedTensorType>();

    // Cast I32 offsets into I64
    SmallVector<Value> i64Offsets;
    for (auto offset : op.getOffsets()) {
      auto i64Offset = builder.create<arith::ExtSIOp>(
          op.getLoc(), builder.getI64Type(), offset);
      i64Offsets.push_back(i64Offset);
    }

    // Save information
    rewritedInfo[op.getResult()] =
        RewritedInfo(op.getBase(), op.getShape(), op.getStrides(), i64Offsets,
                     tensorType.getShape(), tensorType.getEncoding());

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteAdvanceOp(OpBuilder &builder, tt::AdvanceOp op,
                              std::stack<Operation *> &eraser,
                              const DenseSet<Value> &valueToRemove) {
    if (!valueToRemove.count(op.getResult())) {
      return nullptr;
    }
    // Get info from previous results
    assert(rewritedInfo.count(op.getPtr()));
    auto info = rewritedInfo[op.getPtr()];

    // Calculate new offsets
    assert(info.length() == op.getOffsets().size());
    SmallVector<Value> newOffsets;
    for (int i = 0; i < info.length(); ++i) {
      Value i64Offset = builder.create<arith::ExtSIOp>(
          op.getLoc(), builder.getI64Type(), op.getOffsets()[i]);
      Value newOffset = builder.create<arith::AddIOp>(
          op.getLoc(), info.getOffset(i), i64Offset);
      newOffsets.push_back(newOffset);
    }

    // Save info for later use
    info.setOffsets(newOffsets);
    rewritedInfo[op.getResult()] = info;

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteLoadStoreOp(OpBuilder &builder, Operation *op,
                                std::stack<Operation *> &eraser,
                                const DenseSet<Value> &valueToRemove) {
    if (!valueToRemove.count(op->getOperand(0)))
      return nullptr;

    // We only have to rewrite load/stores with tensor pointers
    auto ptr = op->getOperand(0);
    if (!tt::isTensorPointerType(ptr.getType()))
      return nullptr;

    // Get info from previous results
    assert(rewritedInfo.count(ptr));
    auto info = rewritedInfo[ptr];

    // Load/store with tensor pointers implicitly will check the bound while
    // accessing memory, so we should set `mask` and `other` (according to the
    // padding). Also note that load with tensor pointers do not have `mask` and
    // `other` while building IR from Python AST
    std::optional<ArrayRef<int>> boundaryCheck;
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      assert(!loadOp.getMask() && !loadOp.getOther());
      boundaryCheck = loadOp.getBoundaryCheck();
      if (auto valueType =
              dyn_cast<RankedTensorType>(loadOp.getResult().getType()))
        info.setEncoding(valueType.getEncoding());
    } else if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      assert(!storeOp.getMask());
      boundaryCheck = storeOp.getBoundaryCheck();
      if (auto valueType =
              dyn_cast<RankedTensorType>(storeOp.getValue().getType()))
        info.setEncoding(valueType.getEncoding());
    }

    // Generate new `ptr`, `mask` and `other`
    auto newPtr = info.generatePtr(builder, op->getLoc());
    auto newMask = info.generateMask(builder, op->getLoc(), boundaryCheck);
    Value newOther;
    if (auto loadOp = dyn_cast<tt::LoadOp>(op))
      newOther = info.generateOther(builder, op->getLoc(), loadOp.getPadding());

    // Create a new operation
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      auto newResult = builder.create<tt::LoadOp>(
          loadOp.getLoc(), loadOp.getResult().getType(), newPtr, newMask,
          newOther, loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      op->getResult(0).replaceAllUsesWith(newResult);
    } else if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      builder.create<tt::StoreOp>(storeOp.getLoc(), newPtr, storeOp.getValue(),
                                  newMask, storeOp.getCache(),
                                  storeOp.getEvict());
    }

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteForOp(OpBuilder &builder, scf::ForOp op,
                          std::stack<Operation *> &eraser,
                          DenseSet<Value> &valueToRemove) {
    // Generate new iteration operands and set rewrited information
    SmallVector<Value> oldIterOperands = op.getIterOperands();
    SmallVector<Value> newIterOperands = op.getIterOperands();
    for (unsigned i = 0, oldI = 0, size = op.getNumIterOperands(); i < size;
         ++i, ++oldI) {
      if (!tt::isTensorPointerType(newIterOperands[i].getType()))
        continue;
      if (!valueToRemove.count(newIterOperands[i]))
        continue;

      // Expand the tensor pointer into offsets
      assert(rewritedInfo.count(newIterOperands[i]));
      auto info = rewritedInfo[newIterOperands[i]];
      newIterOperands =
          generateNewOperands(newIterOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }

    // Rebuild the loop type
    auto newForOp = builder.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                               op.getUpperBound(), op.getStep(),
                                               newIterOperands);

    // Create value mapping. Note that for tensor pointers, we use identity
    // mapping. It may refer to a value in the old loop, but we will rewrite it
    // later
    IRMapping mapping;
    for (unsigned i = 0, oldI = 0; oldI < op.getNumIterOperands();
         ++i, ++oldI) {
      auto oldRegionIterArg = op.getRegionIterArg(oldI);
      if (tt::isTensorPointerType(oldRegionIterArg.getType()) &&
          valueToRemove.count(oldIterOperands[oldI])) {
        // Pass rewrited info inside
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        mapping.map(oldRegionIterArg, oldRegionIterArg);
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getRegionIterArg(i + j));
        rewritedInfo[oldRegionIterArg] = info;
        i += info.length() - 1;
      } else {
        mapping.map(oldRegionIterArg, newForOp.getRegionIterArg(i));
      }
    }
    mapping.map(op.getInductionVar(), newForOp.getInductionVar());

    // Clone body
    builder.setInsertionPointToStart(newForOp.getBody());
    for (Operation &opInFor : *op.getBody()) {
      Operation *newOp = builder.clone(opInFor, mapping);
      for (unsigned i = 0; i < opInFor.getNumResults(); ++i) {
        if (valueToRemove.count(opInFor.getResult(i)))
          valueToRemove.insert(newOp->getResult(i));
        mapping.map(opInFor.getResult(i), newOp->getResult(i));
      }
    }

    // supported nested scf.for ops
    for (auto &[k, v] : mapping.getValueMap())
      if (valueToRemove.find(k) != valueToRemove.end())
        valueToRemove.insert(v);

    // Replace later usages
    assert(op.getNumResults() == op.getNumIterOperands());
    for (unsigned i = 0, oldI = 0; oldI < op.getNumResults(); ++i, ++oldI) {
      auto oldResult = op.getResult(oldI);
      if (tt::isTensorPointerType(oldResult.getType()) &&
          valueToRemove.count(oldIterOperands[oldI])) {
        // Pack new offsets into rewrited info
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getResult(i + j));
        i += info.length() - 1;
        rewritedInfo[oldResult] = info;
      } else {
        oldResult.replaceAllUsesWith(newForOp.getResult(i));
      }
    }

    // Erase later
    eraser.push(op);
    return newForOp;
  }

  Operation *rewriteYieldOp(OpBuilder &builder, scf::YieldOp op,
                            std::stack<Operation *> &eraser,
                            const DenseSet<Value> &valueToRemove) {
    // Replace tensor pointers with offsets
    SmallVector<Value> newOperands = op->getOperands();
    for (unsigned i = 0, size = op.getNumOperands(); i < size; ++i) {
      if (!tt::isTensorPointerType(newOperands[i].getType()))
        continue;
      if (!valueToRemove.count(newOperands[i]))
        continue;

      assert(rewritedInfo.count(newOperands[i]));
      auto info = rewritedInfo[newOperands[i]];
      newOperands = generateNewOperands(newOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }
    op->setOperands(newOperands);

    // No need to erase
    return nullptr;
  }

  Operation *rewriteIfOp(OpBuilder &builder, scf::IfOp op,
                         std::stack<Operation *> &eraser,
                         DenseSet<Value> &valueToRemove) {
    auto thenYieldOp = op.thenYield();
    assert(op.getNumResults() == thenYieldOp.getNumOperands());
    SmallVector<Value> results = thenYieldOp.getOperands();

    // get new result types
    SmallVector<Type> newRetTypes;
    for (unsigned i = 0; i < results.size(); ++i) {
      if (!tt::isTensorPointerType(results[i].getType()) ||
          !valueToRemove.count(results[i])) {
        newRetTypes.push_back(results[i].getType());
        continue;
      }
      auto makeTensorPtrOp = getMakeTensorPtrOp(results[i]);
      assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
      auto info = rewritedInfo[makeTensorPtrOp.getResult()];
      for (unsigned j = 0; j < info.length(); ++j) {
        newRetTypes.push_back(builder.getI64Type());
      }
    }

    // create and clone new IfOp
    bool hasElse = !op.getElseRegion().empty();
    scf::IfOp newOp = builder.create<scf::IfOp>(op.getLoc(), newRetTypes,
                                                op.getCondition(), hasElse);
    IRMapping mapping;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      mapping.map(op->getOperand(i), newOp->getOperand(i));
    }
    auto rematerialize = [&](Block *block) {
      for (Operation &opInIf : block->getOperations()) {
        auto newOp = builder.clone(opInIf, mapping);
      }
    };
    builder.setInsertionPointToStart(newOp.thenBlock());
    rematerialize(op.thenBlock());
    if (hasElse) {
      builder.setInsertionPointToStart(newOp.elseBlock());
      rematerialize(op.elseBlock());
    }

    // supported nested ops
    for (auto &[k, v] : mapping.getValueMap())
      if (valueToRemove.find(k) != valueToRemove.end())
        valueToRemove.insert(v);

    // update rewritedInfo
    unsigned oldResIdx = 0, newResIdx = 0;
    while (oldResIdx < results.size()) {
      if (!tt::isTensorPointerType(results[oldResIdx].getType()) ||
          !valueToRemove.count(results[oldResIdx])) {
        oldResIdx++;
        newResIdx++;
      } else {
        auto makeTensorPtrOp = getMakeTensorPtrOp(results[oldResIdx]);
        assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
        auto info = rewritedInfo[makeTensorPtrOp.getResult()];
        for (unsigned j = 0; j < info.length(); ++j) {
          info.setOffset(j, newOp->getResult(newResIdx++));
        }
        rewritedInfo[op.getResult(oldResIdx)] = info;
        oldResIdx++;
      }
    }

    eraser.push(op);
    return newOp;
  }

  Operation *rewriteOp(Operation *op, std::stack<Operation *> &eraser,
                       DenseSet<Value> &valueToRemove) {
    OpBuilder builder(op);

    // Rewrite `make_tensor_ptr` and `advance` and make a tensor of pointers
    // Rewriting functions return the next operation to visit, if there is no
    // next one, simply return `nullptr`
    std::pair<Value, RewritedInfo> rewrited;
    if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
      return rewriteMakeTensorPtrOp(builder, makeTensorPtrOp, eraser,
                                    valueToRemove);
    } else if (auto advanceOp = dyn_cast<tt::AdvanceOp>(op)) {
      return rewriteAdvanceOp(builder, advanceOp, eraser, valueToRemove);
    } else if (isa<tt::LoadOp>(op) || isa<tt::StoreOp>(op)) {
      return rewriteLoadStoreOp(builder, op, eraser, valueToRemove);
    } else if (op->getDialect()->getNamespace() == "scf" ||
               op->getDialect()->getNamespace() == "cf") {
      if (!needRewrite(op, valueToRemove))
        return op;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return rewriteForOp(builder, forOp, eraser, valueToRemove);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        return rewriteYieldOp(builder, yieldOp, eraser, valueToRemove);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        return rewriteIfOp(builder, ifOp, eraser, valueToRemove);
      } else {
        llvm_unreachable("Currently we only support tensor pointer usages "
                         "inside a `scf::ForOp` or `scf::IfOp`, others such as "
                         "`scf::WhileOp`, `cf::BranchOp` or `cf::CondBranchOp` "
                         "are not supported yet");
      }
    }

    // Otherwise return the original one
    return op;
  }

  void visitOperation(Operation *op, std::stack<Operation *> &eraser,
                      DenseSet<Value> &valueToRemove) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        // We need an extra copy because erasing operations may break the
        // iterator behavior
        SmallVector<Operation *> blockCopy;
        for (auto &nestedOp : block)
          blockCopy.push_back(&nestedOp);

        // Rewrite and recursively visit
        for (auto &nestedOp : blockCopy) {
          if (auto newOp = rewriteOp(nestedOp, eraser, valueToRemove))
            visitOperation(newOp, eraser, valueToRemove);
        }
      }
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    DenseSet<Value> valueToRemove;
    mod.walk([&valueToRemove,
              computeCapability = this->computeCapability](Operation *op) {
      if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        if (shouldRemove(makeTensorPtrOp, computeCapability))
          valueToRemove.insert(op->getResult(0));
      }
      if (llvm::isa<tt::AdvanceOp>(op)) {
        auto src = op->getOperand(0);
        if (tt::isTensorPointerType(src.getType())) {
          auto makeTensorPtrOp = getMakeTensorPtrOp(src);
          if (shouldRemove(makeTensorPtrOp, computeCapability)) {
            valueToRemove.insert(op->getResult(0));
          }
        }
      }
      if (llvm::isa<tt::LoadOp, tt::StoreOp>(op)) {
        auto src = op->getOperand(0);
        if (tt::isTensorPointerType(src.getType())) {
          auto makeTensorPtrOp = getMakeTensorPtrOp(src);
          if (shouldRemove(makeTensorPtrOp, computeCapability))
            valueToRemove.insert(src);
        }
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        SmallVector<Value> iterOperands = forOp.getIterOperands();
        for (unsigned i = 0, size = forOp.getNumIterOperands(); i < size; ++i) {
          if (tt::isTensorPointerType(iterOperands[i].getType())) {
            auto makeTensorPtrOp = getMakeTensorPtrOp(iterOperands[i]);
            if (shouldRemove(makeTensorPtrOp, computeCapability))
              valueToRemove.insert(iterOperands[i]);
          }
        }
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        SmallVector<Value> operands = yieldOp->getOperands();
        for (unsigned i = 0, size = yieldOp.getNumOperands(); i < size; ++i) {
          if (tt::isTensorPointerType(operands[i].getType())) {
            auto makeTensorPtrOp = getMakeTensorPtrOp(operands[i]);
            if (shouldRemove(makeTensorPtrOp, computeCapability))
              valueToRemove.insert(operands[i]);
          }
        }
      }
    });

    // NOTES(Chenggang): we don't use `ConversionPatternRewriter`, because
    // MLIR does not support one-multiple value mapping. For example, if we use
    // `ConversionPatternRewriter`, we can not make a type converter, which
    // converts `ptr<tensor>` into multiple types `ptr<>, int64, int64, ...`
    // (containing the base/offsets/strides...). What we can do is to convert
    // `ptr<tensor>` into a single type `Tuple<ptr<>, int64, int64, ...>`. But
    // in this way, we also have to define `PackTuple` and `UnpackTuple`
    // operations and make a canonicalization pass to optimize, which is much
    // So here we recursively build the IR, to be specific, we have to rewrite
    // `tt.make_tensor_ptr`, `tt.advance`, `tt.load`, `tt.store`,
    // `scf.for` (tensor pointer usages may be in a loop fashion)
    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser, valueToRemove);

    // The operation could not be erased during visit, because they may have
    // later usages, so we erase after visit
    rewritedInfo.clear();
    valueToRemove.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonGPURewriteTensorPointerPass(int computeCapability) {
  return std::make_unique<TritonGPURewriteTensorPointerPass>(computeCapability);
}
