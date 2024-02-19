add_mlir_dialect_library(TritonCPUTransforms
  FuseOperation.cpp
  PlanCTA.cpp

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  MLIRTransformUtils
)
