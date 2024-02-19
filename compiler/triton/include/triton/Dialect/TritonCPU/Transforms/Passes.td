#ifndef TRITONCPU_PASSES
#define TRITONCPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonCPUPipeline : Pass<"tritongpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Replace `LoadOp` in loops by `InsertSliceAsyncOp` instructions that asynchronously construct the data
    needed at the next iteration
  }];

  let constructor = "mlir::createTritonCPUPipelinePass()";

  let dependentDialects = ["mlir::triton::cpu::TritonCPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaCPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"numWarps", "num-warps",
           "int32_t", /*default*/"4",
           "number of warps per block">,
    Option<"numCTAs", "num-ctas",
           "int32_t", /*default*/"1",
           "number of CTAs per CGA">,
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">
  ];
}

// todo: fixme
def TritonCPUPrefetch : Pass<"tritongpu-prefetch", "mlir::ModuleOp"> {
  let summary = "prefetch";

  let description = [{
    Decompose `DotOp` instructions in loops into several finer-grained `DotOp`
    that may have their operands constructed at the end of the previous iteration
  }];

  let constructor = "mlir::createTritonCPUPrefetchPass()";

  let dependentDialects = ["mlir::triton::cpu::TritonCPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}


def TritonCPUBlockDecomposition : Pass<"tritoncpu-block-decomposition", "mlir::ModuleOp"> {
  let summary = "block-decomposition";

  let description = [{
    block-decomposition
  }];

  let constructor = "mlir::createTritonCPUBlockDecompositionPass()";

  let dependentDialects = ["mlir::triton::cpu::TritonCPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonCPUInterBlockReordering : Pass<"tritoncpu-inter-block-reordering", "mlir::ModuleOp"> {
  let summary = "inter-block-reordering";

  let description = [{
    inter-block-reordering
  }];

  let constructor = "mlir::createTritonCPUInterBlockReordering()";

  let dependentDialects = ["mlir::triton::cpu::TritonCPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}


def TritonCPUIntraBlockScheduling : Pass<"tritoncpu-intra-block-scheduling", "mlir::ModuleOp"> {
  let summary = "intra-block-scheduling";

  let description = [{
    intra-block-scheduling
  }];

  let constructor = "mlir::createTritonCPUIntraBlockScheduling()";

  let dependentDialects = ["mlir::triton::cpu::TritonCPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

#endif
