; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-- -mattr=fma | FileCheck %s

; Propagation of IR FMF should not drop flags when adding the DAG reduction flag.
; This should include an FMA instruction, not separate FMUL/FADD.

define double @julia_dotf(<4 x double> %x, <4 x double> %y, <4 x double> %z, i1 %t3) {
; CHECK-LABEL: julia_dotf:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vfmadd213pd {{.*#+}} ymm0 = (ymm1 * ymm0) + ymm2
; CHECK-NEXT:    vextractf128 $1, %ymm0, %xmm1
; CHECK-NEXT:    vaddpd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vpermilpd {{.*#+}} xmm1 = xmm0[1,0]
; CHECK-NEXT:    vaddsd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %t1 = fmul contract <4 x double> %x, %y
  %t2 = fadd fast <4 x double> %z, %t1
  %rdx.shuf = shufflevector <4 x double> %t2, <4 x double> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx22 = fadd fast <4 x double> %t2, %rdx.shuf
  %rdx.shuf23 = shufflevector <4 x double> %bin.rdx22, <4 x double> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx24 = fadd fast <4 x double> %bin.rdx22, %rdx.shuf23
  %t4 = extractelement <4 x double> %bin.rdx24, i32 0
  ret double %t4
}
