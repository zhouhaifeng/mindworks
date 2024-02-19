#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/Transforms/Passes.h"
#include <queue>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonCPU/Transforms/Passes.h.inc"

class FuseOperations : public mlir::RewritePattern {
public:
  explicit FuseOperations(mlir::MLIRContext *context)
      : mlir::RewritePattern(2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Check if the operation is the first operation in the sequence
    auto firstOp = dyn_cast<FirstOpType>(op);
    if (!firstOp)
      return mlir::failure();

    // Check if the result of the first operation is used by the second operation
    auto secondOp = dyn_cast<SecondOpType>(firstOp.getResult().getUsers().begin());
    if (!secondOp)
      return mlir::failure();

    // Replace the two operations with a fused operation
    rewriter.replaceOpWithNewOp<FusedOpType>(op, /* arguments */);
    return mlir::success();
  }
};