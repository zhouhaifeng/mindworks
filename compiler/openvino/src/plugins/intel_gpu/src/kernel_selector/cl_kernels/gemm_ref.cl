// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

// Required JIT definitions:
// TRANSPOSE_INPUT0 [1/0]      - whether to tranpose first input.
// TRANSPOSE_INPUT1 [1/0]      - whether to tranpose second input.
// ACCUMULATOR_TYPE [DataType] - type used for intermediate results accumulation.

inline uint FUNC(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, y, x);
#else
#if INPUT0_DIMS == 4
    return INPUT0_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 0 format
#endif
#endif
}

inline uint FUNC(get_input0_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if !TRANSPOSE_INPUT0
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#else
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, x, y);
#endif
}

inline uint FUNC(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, y, x);
#else
#if INPUT1_DIMS == 4
    return INPUT1_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT1_DIMS == 5
    return INPUT1_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT1_DIMS == 6
    return INPUT1_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input1_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if !TRANSPOSE_INPUT1
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#else
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, x, y);
#endif
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, y, x);
#else
#if INPUT2_DIMS == 4
    return INPUT2_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT2_DIMS == 5
    return INPUT2_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT2_DIMS == 6
    return INPUT2_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 2 format
#endif
#endif
}
#endif // INPUT2_TYPE

KERNEL(gemm_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* input2,
#endif
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x = (uint)get_global_id(0);
    const uint y = (uint)get_global_id(1);

    uint bidx = get_global_id(2);
    const uint b = bidx % OUTPUT_BATCH_NUM;
    bidx /= OUTPUT_BATCH_NUM;
    const uint f = bidx % OUTPUT_FEATURE_NUM;
    bidx /= OUTPUT_FEATURE_NUM;
    const uint z = bidx % OUTPUT_SIZE_Z;
    bidx /= OUTPUT_SIZE_Z;
    const uint w = bidx % OUTPUT_SIZE_W;

#if !TRANSPOSE_INPUT0
    const uint K = INPUT0_SIZE_X;
#else
    const uint K = INPUT0_SIZE_Y;
#endif

    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;

    for (uint ki = 0; ki < K; ++ki) {
        uint in0_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, ki);
        uint in1_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, ki, x);

        ACCUMULATOR_TYPE val0 = TO_ACCUMULATOR_TYPE(input0[in0_idx]);
        ACCUMULATOR_TYPE val1 = TO_ACCUMULATOR_TYPE(input1[in1_idx]);

        acc += val0 * val1;
    }

    acc = TO_ACCUMULATOR_TYPE(ALPHA) * acc;

#ifdef INPUT2_TYPE
    {
        uint in2_idx = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
        ACCUMULATOR_TYPE val2 = TO_ACCUMULATOR_TYPE(input2[in2_idx]);

        acc += TO_ACCUMULATOR_TYPE(BETA) * val2;
    }
#endif

    const uint dst_index = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);

    ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(acc);

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[dst_index] = res;
#else
    output[dst_index] = dequantized;
#endif
}
