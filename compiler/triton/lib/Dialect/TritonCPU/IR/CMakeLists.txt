add_mlir_dialect_library(TritonCPUIR
  Dialect.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TritonCPUTableGen
  TritonCPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRCPUOps
  TritonIR
)
