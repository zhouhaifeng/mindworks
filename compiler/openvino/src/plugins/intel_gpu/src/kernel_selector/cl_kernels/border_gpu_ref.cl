// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

KERNEL(border_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#ifdef BEGIN_TYPE
    const __global BEGIN_TYPE* begin,
#endif
#ifdef END_TYPE
    const __global END_TYPE* end,
#endif
#ifdef BORDER_VALUE_TYPE
    const __global BORDER_VALUE_TYPE* border_value_param,
#endif
    __global OUTPUT_TYPE* output)
{
#ifdef BEGIN_TYPE
    const uint begin_b = begin[0];
    const uint begin_f = begin[1];
    uint begin_offset = 2;
    #if INPUT0_DIMS == 6
    const uint begin_w = begin[begin_offset];
    begin_offset += 1;
    #endif
    #if INPUT0_DIMS >= 5
    const uint begin_z = begin[begin_offset];
    begin_offset += 1;
    #endif
    const uint begin_y = begin[begin_offset];
    const uint begin_x = begin[begin_offset + 1];
#else
    const uint begin_b = LT_SIZES_BATCH_NUM;
    const uint begin_f = LT_SIZES_FEATURE_NUM;
    #if INPUT0_DIMS == 6
    const uint begin_w = LT_SIZES_SIZE_W;
    #endif
    #if INPUT0_DIMS >= 5
    const uint begin_z = LT_SIZES_SIZE_Z;
    #endif
    const uint begin_y = LT_SIZES_SIZE_Y;
    const uint begin_x = LT_SIZES_SIZE_X;
#endif

#ifdef END_TYPE
    const uint end_b = end[0];
    const uint end_f = end[1];
    uint end_offset = 2;
    #if INPUT0_DIMS == 6
    const uint end_w = end[end_offset];
    end_offset += 1;
    #endif
    #if INPUT0_DIMS >= 5
    const uint end_z = end[end_offset];
    end_offset += 1;
    #endif
    const uint end_y = end[end_offset];
    const uint end_x = end[end_offset + 1];
#else
    const uint end_b = RB_SIZES_BATCH_NUM;
    const uint end_f = RB_SIZES_FEATURE_NUM;
    #if INPUT0_DIMS == 6
    const uint end_w = RB_SIZES_SIZE_W;
    #endif
    #if INPUT0_DIMS >= 5
    const uint end_z = RB_SIZES_SIZE_Z;
    #endif
    const uint end_y = RB_SIZES_SIZE_Y;
    const uint end_x = RB_SIZES_SIZE_X;
#endif

    // [CONSTEXPR]
    // Border sizes(left-top):
    const uint blt_sb = begin_b;
    const uint blt_sf = begin_f;
    const uint blt_sy = begin_y;
    const uint blt_sx = begin_x;
#if INPUT0_DIMS == 6
    const uint blt_sw = begin_w;
#else
    const uint blt_sw = 0;
#endif
#if INPUT0_DIMS >= 5
    const uint blt_sz = begin_z;
#else
    const uint blt_sz = 0;
#endif

    // Border sizes(right-bottom):
    const uint brb_sb = end_b;
    const uint brb_sf = end_f;
    const uint brb_sy = end_y;
    const uint brb_sx = end_x;
#if INPUT0_DIMS == 6
    const uint brb_sw = end_w;
#else
    const uint brb_sw = 0;
#endif
#if INPUT0_DIMS >= 5
    const uint brb_sz = end_z;
#else
    const uint brb_sz = 0;
#endif

    // Input sizes:
    const uint in_sx = INPUT0_SIZE_X;
    const uint in_sy = INPUT0_SIZE_Y;
    const uint in_sz = INPUT0_SIZE_Z;
    const uint in_sw = INPUT0_SIZE_W;
    const uint in_sf = INPUT0_FEATURE_NUM;
    const uint in_sb = INPUT0_BATCH_NUM;

    // Input limits (exclusive; tested on output position):
    const uint in_lx = in_sx + blt_sx;
    const uint in_ly = in_sy + blt_sy;
    const uint in_lz = in_sz + blt_sz;
    const uint in_lw = in_sw + blt_sw;
    const uint in_lf = in_sf + blt_sf;
    const uint in_lb = in_sb + blt_sb;

    const uint out_xz  = (uint) get_global_id(0);
    const uint out_yw  = (uint) get_global_id(1);
    const uint out_fb = (uint) get_global_id(2);

    const uint out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const uint out_b  = out_fb / OUTPUT_FEATURE_NUM;

    const uint out_x  = out_xz % OUTPUT_SIZE_X;
    const uint out_z  = out_xz / OUTPUT_SIZE_X;

    const uint out_y  = out_yw % OUTPUT_SIZE_Y;
    const uint out_w  = out_yw / OUTPUT_SIZE_Y;

#ifdef BORDER_TYPE_CONSTANT
    #ifdef BORDER_VALUE_TYPE
    INPUT0_TYPE in_val = TO_INPUT0_TYPE(border_value_param[0]);
    #else
    INPUT0_TYPE in_val = TO_INPUT0_TYPE(BORDER_VALUE);
    #endif

    if (out_x >= blt_sx & out_x < in_lx &
        out_y >= blt_sy & out_y < in_ly &
        out_z >= blt_sz & out_z < in_lz &
        out_w >= blt_sw & out_w < in_lw &
        out_f >= blt_sf & out_f < in_lf &
        out_b >= blt_sb & out_b < in_lb)
    {
        const uint in_x = out_x - blt_sx;
        const uint in_y = out_y - blt_sy;
        const uint in_z = out_z - blt_sz;
        const uint in_w = out_w - blt_sw;
        const uint in_f = out_f - blt_sf;
        const uint in_b = out_b - blt_sb;

        const uint in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
        in_val = input[in_pos];
    }
#elif defined BORDER_TYPE_EDGE
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? 0 : in_sx - 1);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? 0 : in_sy - 1);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? 0 : in_sz - 1);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? 0 : in_sw - 1);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? 0 : in_sf - 1);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? 0 : in_sb - 1);

    const uint in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
    INPUT0_TYPE in_val = input[in_pos];
#elif defined BORDER_TYPE_MIRROR
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - 1 - out_x : in_sx + in_lx - 1 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - 1 - out_y : in_sy + in_ly - 1 - out_y);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? blt_sz - 1 - out_z : in_sz + in_lz - 1 - out_z);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? blt_sw - 1 - out_w : in_sw + in_lw - 1 - out_w);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - 1 - out_f : in_sf + in_lf - 1 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - 1 - out_b : in_sb + in_lb - 1 - out_b);

    const uint in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
    INPUT0_TYPE in_val = input[in_pos];
#elif defined BORDER_TYPE_MIRROR_101
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - out_x : in_sx + in_lx - 2 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - out_y : in_sy + in_ly - 2 - out_y);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? blt_sz - out_z : in_sz + in_lz - 2 - out_z);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? blt_sw - out_w : in_sw + in_lw - 2 - out_w);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - out_f : in_sf + in_lf - 2 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - out_b : in_sb + in_lb - 2 - out_b);

    const uint in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
    INPUT0_TYPE in_val = input[in_pos];
#else
    #error Unsupported border type.
#endif

    const uint out_pos = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_w, out_z, out_y, out_x);
    output[out_pos] = in_val;
}
