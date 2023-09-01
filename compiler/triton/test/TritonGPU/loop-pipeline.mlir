// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 1]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 1]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

// CHECK: tt.func @matmul_loop
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[LOOP_COND_0:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
// CHECK: %[[ABUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK: %[[A0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[LOOP_COND_0_SPLAT_A]]
// CHECK: %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK: %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[LOOP_COND_0_SPLAT_B]]
// CHECK-DAG: %[[IV_1:.*]] = arith.addi %[[LB]], %[[STEP:.*]]
// CHECK-DAG: %[[LOOP_COND_1:.*]] = arith.cmpi slt, %[[IV_1]], %[[UB]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK: %[[A1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[LOOP_COND_1_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK: %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[LOOP_COND_1_SPLAT_B]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK: %[[A0:.*]] = triton_gpu.extract_slice %[[A1BUFFER]][0, 0, 0]
// CHECK: %[[B0:.*]] = triton_gpu.extract_slice %[[B1BUFFER]][0, 0, 0]
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_0]]
// CHECK:   %[[arg_a0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_a0]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = triton_gpu.convert_layout %[[arg_b0]]
// CHECK:   %[[arg_b0_dot_op_1:.*]] = arith.mulf %[[arg_b0_dot_op_0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_1]], {{.*}}
// CHECK-DAG: %[[NEXT_LOOP_IDX:.*]] = arith.addi %[[LOOP_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_LOOP:.*]] = arith.cmpi uge, %[[NEXT_LOOP_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_IDX:.*]] = arith.select %[[CMP_LOOP]], %[[CONSTANT_0]], %[[NEXT_LOOP_IDX]]
// CHECK:   %[[NEXT_A_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[PIPELINE_IDX]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[PIPELINE_IDX]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK:   %[[NEXT_A:.*]] = triton_gpu.extract_slice %[[NEXT_A_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.extract_slice %[[NEXT_B_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK-DAG: %[[PIPELINE_IDX_PLUS_ONE:.*]] = arith.addi %[[PIPELINE_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[CMP_PIPELINE:.*]] = arith.cmpi uge, %[[PIPELINE_IDX_PLUS_ONE]], %[[CONSTANT_3]]
// CHECK-DAG: %[[NEXT_PIPELINE_IDX:.*]] = arith.select %[[CMP_PIPELINE]], %[[CONSTANT_0]], %[[PIPELINE_IDX_PLUS_ONE]]
// CHECK:   scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_BUFFER]], %[[NEXT_B_BUFFER]], %[[NEXT_A]], %[[NEXT_B]], {{.*}}, {{.*}}, {{.*}}, %[[NEXT_PIPELINE_IDX]], %[[EXTRACT_IDX]]
tt.func @matmul_loop(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>


  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %b_scale = arith.constant dense<4.> : tensor<32x128xf16, #B>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b_ = triton_gpu.convert_layout %b__ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK: tt.func @matmul_loop_nested
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK: scf.for
// CHECK:   %[[ABUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK:   %[[A0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK:   %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK:   %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK:   %[[A1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK:   %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK:   %[[A0:.*]] = triton_gpu.extract_slice %[[A1BUFFER]][0, 0, 0]
// CHECK:   %[[B0:.*]] = triton_gpu.extract_slice %[[B1BUFFER]][0, 0, 0]
// CHECK:   scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_0]]
// CHECK:     %[[arg_a0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_a0]]
// CHECK:     %[[arg_b0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_b0]]
// CHECK:     tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op]], {{.*}}
// CHECK-DAG: %[[NEXT_LOOP_IDX:.*]] = arith.addi %[[LOOP_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_LOOP:.*]] = arith.cmpi uge, %[[NEXT_LOOP_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_IDX:.*]] = arith.select %[[CMP_LOOP]], %[[CONSTANT_0]], %[[NEXT_LOOP_IDX]]
// CHECK:     %[[NEXT_A_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[PIPELINE_IDX]]
// CHECK:     %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[PIPELINE_IDX]]
// CHECK:     triton_gpu.async_wait {num = 2 : i32}
// CHECK:   %[[NEXT_A:.*]] = triton_gpu.extract_slice %[[NEXT_A_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.extract_slice %[[NEXT_B_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK-DAG: %[[PIPELINE_IDX_PLUS_ONE:.*]] = arith.addi %[[PIPELINE_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[CMP_PIPELINE:.*]] = arith.cmpi uge, %[[PIPELINE_IDX_PLUS_ONE]], %[[CONSTANT_3]]
// CHECK-DAG: %[[NEXT_PIPELINE_IDX:.*]] = arith.select %[[CMP_PIPELINE]], %[[CONSTANT_0]], %[[PIPELINE_IDX_PLUS_ONE]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_BUFFER]], %[[NEXT_B_BUFFER]], %[[NEXT_A]], %[[NEXT_B]], {{.*}}, {{.*}}, {{.*}}, %[[NEXT_PIPELINE_IDX]], %[[EXTRACT_IDX]]
tt.func @matmul_loop_nested(%lb : index, %ub : index, %step : index,
                         %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                         %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C>{

  %c_start = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %loop1:1 = scf.for %iv0 = %lb to %ub step %step iter_args(%c_init = %c_start) -> (tensor<128x128xf32, #C>) {
    // A ptrs
    %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
    %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
    %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // B ptrs
    %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
    %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
    %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
    %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
    %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    %loop2:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
      %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
      %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
      %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }

    scf.yield %loop2#2 : tensor<128x128xf32, #C>
  }
  tt.return %loop1#0 : tensor<128x128xf32, #C>
}


// CHECK: tt.func @matmul_loop_single_pipeline
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK: %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK: %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK: %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK: triton_gpu.async_wait {num = 1 : i32}
// CHECK: %[[B0:.*]] = triton_gpu.extract_slice %[[B1BUFFER]][0, 0, 0]
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_0]]
// CHECK:   %[[arg_b0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_b0]]
// CHECK:   tt.dot {{.*}}, %[[arg_b0_dot_op]], {{.*}}
// CHECK-DAG: %[[NEXT_LOOP_IDX:.*]] = arith.addi %[[LOOP_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_LOOP:.*]] = arith.cmpi uge, %[[NEXT_LOOP_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_IDX:.*]] = arith.select %[[CMP_LOOP]], %[[CONSTANT_0]], %[[NEXT_LOOP_IDX]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[PIPELINE_IDX]]
// CHECK:   triton_gpu.async_wait {num = 1 : i32}
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.extract_slice %[[NEXT_B_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK-DAG: %[[PIPELINE_IDX_PLUS_ONE:.*]] = arith.addi %[[PIPELINE_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[CMP_PIPELINE:.*]] = arith.cmpi uge, %[[PIPELINE_IDX_PLUS_ONE]], %[[CONSTANT_3]]
// CHECK-DAG: %[[NEXT_PIPELINE_IDX:.*]] = arith.select %[[CMP_PIPELINE]], %[[CONSTANT_0]], %[[PIPELINE_IDX_PLUS_ONE]]
// CHECK:   scf.yield {{.*}}, {{.*}}, %[[NEXT_B_BUFFER]], %[[NEXT_B]], {{.*}}, {{.*}}, %[[NEXT_PIPELINE_IDX]], %[[EXTRACT_IDX]]
tt.func @matmul_loop_single_pipeline(%lb : index, %ub : index, %step : index,
                                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>

  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#1 : tensor<128x128xf32, #C>
}

// CHECK: tt.func @lut_bmm_scalar
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.async_commit_group
// CHECK: %[[LUT_BUFFER_0:.*]] = tt.load %arg15, {{.*}}
// CHECK: %[[LUT_BUFFER_1:.*]] = arith.muli {{.*}}, %[[LUT_BUFFER_0]]
// CHECK: %[[LUT_BUFFER_2:.*]] = tt.splat %[[LUT_BUFFER_1]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[LUT_BUFFER_2]]
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %arg14, {{.*}}
// CHECK: triton_gpu.insert_slice_async %[[NEXT_BUFFER_1]]
// CHECK: triton_gpu.insert_slice_async %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {num = 2 : i32}
tt.func @lut_bmm_scalar(%77: i64 {tt.divisibility=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: !tt.ptr<i64>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %84 = arith.muli %77, %83 : i64
    %85 = tt.splat %84 : (i64) -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : (tensor<16x16xf16, #BL>) -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK: tt.func @lut_bmm_vector
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.async_commit_group
// CHECK: %[[LUT_BUFFER_0:.*]] = tt.load %arg15, {{.*}}
// CHECK: %[[LUT_BUFFER_1:.*]] = tt.expand_dims %[[LUT_BUFFER_0]] {axis = 1 : i32}
// CHECK: %[[LUT_BUFFER_2:.*]] = tt.broadcast %[[LUT_BUFFER_1]]
// CHECK: %[[LUT_BUFFER_3:.*]] = arith.muli {{.*}}, %[[LUT_BUFFER_2]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[LUT_BUFFER_3]]
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %arg14, {{.*}}
// CHECK: triton_gpu.insert_slice_async %[[NEXT_BUFFER_1]]
// CHECK: triton_gpu.insert_slice_async %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {num = 2 : i32}
tt.func @lut_bmm_vector(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : (i32) -> tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xi64, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: (tensor<16xi64, #BLs1>) -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : (tensor<16x1xi64, #BL>) -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : (tensor<16x16xf16, #BL>) -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK: tt.func @post_load_inv
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK-DAG: %[[IV:.*]] = arith.index_cast
// CHECK: %[[NEXT_IV:.*]] = arith.addi %[[IV]], %c1_i32 : i32
// CHECK-NOT: arith.addi %[[NEXT_IV]]
tt.func @post_load_inv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg3: i32 {tt.divisibility = 16 : i32},
                       %arg4: i32 {tt.divisibility = 16 : i32},
                       %arg5: i32 {tt.divisibility = 16 : i32},
                       %arg6: i32 {tt.divisibility = 16 : i32},
                       %arg7: i32 {tt.divisibility = 16 : i32},
                       %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_index = arith.constant 0 : index
  %c1_index = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %84 = arith.constant 900 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %50 = tt.splat %arg3 : (i32) -> tensor<1x32xi32, #AL>
  %59 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %81 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %66 = tt.splat %arg4 : (i32) -> tensor<32x1xi32, #AL>
  %60 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %82 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %85:3 = scf.for %arg9 = %c0_index to %84 step %c1_index iter_args(%arg10 = %cst, %arg11 = %59, %arg12 = %81) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %130 = arith.index_cast %arg9 : index to i32
    %107 = arith.muli %130, %c32_i32 : i32
    %108 = arith.subi %arg5, %107 : i32
    %109 = tt.splat %108 : (i32) -> tensor<1x32xi32, #AL>
    %110 = "triton_gpu.cmpi"(%50, %109) <{predicate = 2 : i64}> : (tensor<1x32xi32, #AL>, tensor<1x32xi32, #AL>) -> tensor<1x32xi1, #AL>
    %111 = tt.broadcast %110 : (tensor<1x32xi1, #AL>) -> tensor<32x32xi1, #AL>
    %112 = tt.load %arg11, %111, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %113 = tt.splat %108 : (i32) -> tensor<32x1xi32, #AL>
    %114 = "triton_gpu.cmpi"(%66, %113) <{predicate = 2 : i64}> : (tensor<32x1xi32, #AL>, tensor<32x1xi32, #AL>) -> tensor<32x1xi1, #AL>
    %115 = tt.broadcast %114 : (tensor<32x1xi1, #AL>) -> tensor<32x32xi1, #AL>
    %116 = tt.load %arg12, %115, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %117 = triton_gpu.convert_layout %112 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %118 = triton_gpu.convert_layout %116 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %119 = tt.dot %117, %118, %arg10 {allowTF32 = true} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %131 = arith.index_cast %arg9 : index to i32
    %120 = arith.addi %131, %c1_i32 : i32
    %121 = arith.muli %120, %c32_i32 : i32
    %122 = tt.splat %121 : (i32) -> tensor<32x32xi32, #AL>
    %123 = tt.addptr %60, %122 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %124 = arith.muli %121, %arg7 : i32
    %125 = tt.splat %124 : (i32) -> tensor<32x32xi32, #AL>
    %126 = tt.addptr %82, %125 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %119, %123, %126 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %85#0 : tensor<32x32xf32, #C>
}

// CHECK: tt.func @cross_iter_dep
// CHECK: triton_gpu.async_commit_group
// CHECK: triton_gpu.async_commit_group
// CHECK: triton_gpu.async_commit_group
// CHECK: triton_gpu.async_commit_group
// CHECK: %[[PTR0:.*]] = tt.addptr
// CHECK: %[[PTR1:.*]] = tt.addptr
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[BUF0:.*]] = %[[PTR0]], {{.*}}, %[[BUF1:.*]] = %[[PTR1]]
// CHECK: scf.yield
// CHECK-SAME: %[[BUF0]]
// CHECK-SAME: %[[BUF1]]
tt.func @cross_iter_dep(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg3: i32 {tt.divisibility = 16 : i32},
                        %arg4: i32 {tt.divisibility = 16 : i32},
                        %arg5: i32 {tt.divisibility = 16 : i32},
                        %arg6: i32 {tt.divisibility = 16 : i32},
                        %arg7: i32 {tt.divisibility = 16 : i32},
                        %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_i32 = arith.constant 0 : index
  %118 = arith.constant 32 : index
  %c1_i32 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %78 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %110 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %112 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %113 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %116 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %65 = tt.splat %arg3 : (i32) -> tensor<1x32xi32, #AL>
  %88 = tt.splat %arg4 : (i32) -> tensor<32x1xi32, #AL>
  %80 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %119:5 = scf.for %arg9 = %c0_i32 to %118 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %78, %arg12 = %110, %arg13 = %113, %arg14 = %116) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %161 = arith.index_cast %arg9 : index to i32
    %141 = arith.muli %161, %c32_i32 : i32
    %142 = arith.subi %arg5, %141 : i32
    %143 = tt.splat %142 : (i32) -> tensor<1x32xi32, #AL>
    %144 = "triton_gpu.cmpi"(%65, %143) <{predicate = 2 : i64}> : (tensor<1x32xi32, #AL>, tensor<1x32xi32, #AL>) -> tensor<1x32xi1, #AL>
    %145 = tt.broadcast %144 : (tensor<1x32xi1, #AL>) -> tensor<32x32xi1, #AL>
    %146 = tt.load %arg11, %145, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %147 = tt.splat %142 : (i32) -> tensor<32x1xi32, #AL>
    %148 = "triton_gpu.cmpi"(%88, %147) <{predicate = 2 : i64}> : (tensor<32x1xi32, #AL>, tensor<32x1xi32, #AL>) -> tensor<32x1xi1, #AL>
    %149 = tt.broadcast %148 : (tensor<32x1xi1, #AL>) -> tensor<32x32xi1, #AL>
    %150 = tt.load %arg12, %149, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %151 = triton_gpu.convert_layout %146 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %152 = triton_gpu.convert_layout %150 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %153 = tt.dot %151, %152, %arg10 {allowTF32 = true} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %162 = arith.index_cast %arg9 : index to i32
    %154 = arith.addi %162, %c2_i32 : i32
    %155 = arith.muli %154, %c32_i32 : i32
    %156 = tt.splat %155 : (i32) -> tensor<32x32xi32, #AL>
    %157 = tt.addptr %80, %156 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %158 = arith.muli %155, %arg7 : i32
    %159 = tt.splat %158 : (i32) -> tensor<32x32xi32, #AL>
    %160 = tt.addptr %112, %159 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %153, %arg13, %arg14, %157, %160 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %119#0 : tensor<32x32xf32, #C>
}

// CHECK: tt.func @dep_arg_two_uses
// CHECK: tt.expand_dims
// CHECK: tt.expand_dims
// CHECK: tt.expand_dims %arg5
// CHECK-NEXT: tt.expand_dims %arg5
// CHECK: %[[PTR0:.*]] = tt.splat %arg6
// CHECK: %[[PTR1:.*]] = tt.addptr %[[PTR0]]
// CHECK-NEXT: tt.load %[[PTR1]]
tt.func @dep_arg_two_uses(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  %23 = arith.constant 100 : index
  %c64 = arith.constant 64 : i64
  %56 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %57 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %58 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
  %83 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %85 = tt.splat %c64 : (i64) -> tensor<1x32xi64, #AL>
  %86 = tt.splat %c64 : (i64) -> tensor<1x32xi64, #AL>
  %68 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %c32_index = arith.constant 32 : index
  %c32_i32 = arith.index_cast %c32_index : index to i32
  %80 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #BL>
  %88 = arith.truncf %cst_6 : tensor<32x128xf32, #BL> to tensor<32x128xf16, #BL>
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #C>
  %90 = tt.splat %c64 : (i64) -> tensor<32x128xi64, #BL>
  %92 = tt.addptr %arg1, %c32_i32 : !tt.ptr<i32>, i32
  %c0_index = arith.constant 0 : index
  %91:5 = scf.for %arg19 = %c0_index to %23 step %c32_index iter_args(%arg20 = %68, %arg21 = %83, %arg22 = %92, %arg23 = %cst, %arg24 = %80) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>, !tt.ptr<i32>, tensor<128x128xf32, #C>, tensor<32x128x!tt.ptr<f16>, #BL>)   {
    %1750 = arith.subi %23, %arg19 : index
    %175 = arith.index_cast %1750 : index to i32
    %176 = tt.splat %175 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %177 = tt.splat %175 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
    %178 = "triton_gpu.cmpi"(%57, %176) <{predicate = 2 : i64}> : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>) -> tensor<32xi1, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %179 = "triton_gpu.cmpi"(%58, %177) <{predicate = 2 : i64}> : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>, tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>) -> tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #BL}>>
    %180 = tt.expand_dims %178 {axis = 0 : i32} : (tensor<32xi1, #triton_gpu.slice<{dim = 0, parent = #AL}>>) -> tensor<1x32xi1, #AL>
    %181 = tt.expand_dims %179 {axis = 1 : i32} : (tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #BL}>>) -> tensor<32x1xi1, #BL>
    %182 = tt.expand_dims %arg21 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>) -> tensor<1x32xi32, #AL>
    %183 = tt.expand_dims %arg21 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>) -> tensor<1x32xi32, #AL>
    %184 = arith.extsi %182 : tensor<1x32xi32, #AL> to tensor<1x32xi64, #AL>
    %185 = arith.extsi %183 : tensor<1x32xi32, #AL> to tensor<1x32xi64, #AL>
    %186 = arith.muli %184, %85 : tensor<1x32xi64, #AL>
    %187 = arith.muli %185, %86 : tensor<1x32xi64, #AL>
    %188 = tt.broadcast %186 : (tensor<1x32xi64, #AL>) -> tensor<128x32xi64, #AL>
    %189 = tt.broadcast %187 : (tensor<1x32xi64, #AL>) -> tensor<128x32xi64, #AL>
    %190 = tt.addptr %arg20, %188 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi64, #AL>
    %191 = tt.addptr %arg20, %189 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi64, #AL>
    %192 = tt.broadcast %180 : (tensor<1x32xi1, #AL>) -> tensor<128x32xi1, #AL>
    %193 = tt.load %191, %192 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %194 = tt.splat %arg22 : (!tt.ptr<i32>) -> tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %195 = tt.addptr %194, %56 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %196 = tt.load %195 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %197 = tt.addptr %arg22, %c32_i32 : !tt.ptr<i32>, i32
    %198 = tt.broadcast %181 : (tensor<32x1xi1, #BL>) -> tensor<32x128xi1, #BL>
    %199 = tt.load %arg24, %198, %88 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %200 = triton_gpu.convert_layout %193 : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>>
    %201 = triton_gpu.convert_layout %199 : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>>
    %202 = tt.dot %200, %201, %arg23 {allowTF32 = true} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>> -> tensor<128x128xf32, #C>
    %203 = tt.addptr %arg24, %90 : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi64, #BL>
    scf.yield %190, %196, %197, %202, %203 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>, !tt.ptr<i32>, tensor<128x128xf32, #C>, tensor<32x128x!tt.ptr<f16>, #BL>
  }
  tt.return %91#3 : tensor<128x128xf32, #C>
}
