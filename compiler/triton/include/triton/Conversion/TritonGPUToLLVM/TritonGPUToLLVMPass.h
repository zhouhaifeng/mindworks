#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Target/PTX/TmaMetadata.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

//modified by zhf 20040205 begin
enum Target { NVVM, ROCDL, X86, Default = NVVM };
//modified by zhf 20040205 end

#define GEN_PASS_DECL
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(const ConvertTritonGPUToLLVMOptions &options);

} // namespace triton

} // namespace mlir

#endif
