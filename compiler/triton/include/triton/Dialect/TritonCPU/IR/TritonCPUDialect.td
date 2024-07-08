#ifndef TRITONCPU_DIALECT
#define TRITONCPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonCPU_Dialect : Dialect {
  let name = "triton_cpu";

  let cppNamespace = "::mlir::triton::cpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton CPU Dialect.
  }];

  //todo: vector dependentcy and check amx/x86vector dependentcy
  let dependentDialects = [
    "triton::TritonDialect",
    //"mlir::triton::nvgpu::NVGPUDialect",
    //"mlir::gpu::GPUDialect",
    "vector::VectorDialect",
    "x86vector::X86VectorDialect",
    "amx::AMXDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_cpu.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if(!mod->hasAttr("triton_cpu.num-warps"))
        llvm::report_fatal_error(
            "TritonCPU module should contain a triton_cpu.num-warps attribute");
      return mod->getAttr("triton_cpu.num-warps").cast<IntegerAttr>().getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if(!mod->hasAttr("triton_cpu.num-ctas"))
        llvm::report_fatal_error(
            "TritonCPU module should contain a triton_cpu.num-ctas attribute");
      return mod->getAttr("triton_cpu.num-ctas").cast<IntegerAttr>().getInt();
    }
    static int getComputeCapability(ModuleOp mod) {
      if(!mod->hasAttr("triton_cpu.compute-capability"))
        llvm::report_fatal_error(
            "TritonCPU module should contain a triton_gpu.compute-capability attribute");
      return mod->getAttrOfType<IntegerAttr>("triton_cpu.compute-capability").getInt();
    }
    void registerTypes();

    static std::string getThreadsPerWarpAttrName() { return "triton_cpu.threads-per-warp"; }

    static int getThreadsPerWarp(ModuleOp mod) {
      Attribute threadsPerWarp = mod->getDiscardableAttr("triton_cpu.threads-per-warp");
      if(!threadsPerWarp) {
        return 32;
      }
      return threadsPerWarp.cast<IntegerAttr>().getInt();
    }
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
