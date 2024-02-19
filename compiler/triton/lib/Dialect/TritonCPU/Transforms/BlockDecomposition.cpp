#include "triton/Dialect/TritonNvidiaCPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaCPU/Transforms/Passes.h"
#include <queue>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaCPU/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
namespace ttc = ::mlir::triton::cpu;

//type

//attribute

//operation
//todo: fixme: replace `FirstOpType`, `SecondOpType`, and `FusedOpType` with the actual operation types
//todo: fixme: base class `RewritePattern` should be replaced with the actual base class
class BlockDecomposition {
public:
  explicit BlockDecomposition(mlir::MLIRContext *context)
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


} // namespace

std::unique_ptr<Pass>
//todo: fixme: pass arguments
mlir::createTritonNvidiaCPUBlockDecompositionPass(ttc::ClusterInfo *clusterInfo) {
  return std::make_unique<BlockDecompositionPass>(clusterInfo);
}