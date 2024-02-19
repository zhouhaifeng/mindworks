#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

std::unique_ptr<Pass> createTritonCPUPipelinePass(int numStages = 3,
                                                  int numWarps = 4,
                                                  int numCTAs = 1,
                                                  int computeCapability = 80);

std::unique_ptr<Pass>


std::unique_ptr<Pass> createTritonCPUPrefetchPass();

std::unique_ptr<Pass> createTritonCPUBlockDecompositionPass();
std::unique_ptr<Pass> createTritonCPUInterBlockReordering();
std::unique_ptr<Pass> createTritonCPUIntraBlockScheduling();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
