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


struct InterBlockReorderingPass : public TritonCPUInterBlockReorderingBase<InterBlockReorderingPass> {
  InterBlockReorderingPass(ttc::ClusterInfo *clusterInfo) {
    this->clusterInfo = clusterInfo;
  }
  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    OwningRewritePatternList patterns;
    populateTilingPatterns(context, patterns);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass>
//todo: fixme: pass arguments
mlir::createTritonNvidiaCPUInterBlockReorderingPass(ttc::ClusterInfo *clusterInfo) {
  return std::make_unique<BlockDecompositionPass>(clusterInfo);
}