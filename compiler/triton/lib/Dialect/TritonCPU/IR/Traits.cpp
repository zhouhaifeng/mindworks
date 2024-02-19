#include "triton/Dialect/TritonCPU/IR/Traits.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

mlir::LogicalResult
mlir::OpTrait::impl::verifyResultsAreSharedEncoding(Operation *op) {
  if (failed(verifyAtLeastNResults(op, 1)))
    return failure();

  for (auto result : op->getResults())
    if (!triton::cpu::isSharedEncoding(result))
      return op->emitOpError() << "requires all results to be shared encoding";

  return success();
};
