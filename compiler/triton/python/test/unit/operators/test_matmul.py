import itertools

import pytest
import torch

import triton
import triton.language as tl
import triton.ops


def f8_to_f16(x, dtype):

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    dtype = getattr(tl, dtype)
    kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
    return ret


@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, ALLOW_TF32",
    itertools.chain(
        *[
            [
                # 1 warp
                (16, 16, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (32, 16, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (16, 32, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (16, 16, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (32, 16, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (16, 32, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (16, 16, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (64, 16, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (16, 64, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                # 2 warp
                (64, 32, 64, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (32, 64, 64, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (64, 32, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (32, 64, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (128, 32, 32, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (32, 128, 32, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                # 4 warp
                (128, 64, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (64, 128, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (128, 32, 32, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (32, 128, 32, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (128, 32, 64, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (32, 128, 64, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                # 8 warp
                (128, 256, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (256, 128, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                (256, 128, 32, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, True),
                # variable input
                (128, 128, 32, 1, 4, 2, 256, 384, 160, AT, BT, DTYPE, DTYPE, True),
                (128, 128, 32, 1, 4, 2, 107, 233, 128, AT, BT, DTYPE, DTYPE, True),
                (128, 128, 32, 1, 4, 2, 107, 233, 83, AT, BT, DTYPE, DTYPE, True),
                (128, 256, 64, 1, 8, 3, 256, 512, 160, AT, BT, DTYPE, DTYPE, True),
            ] for DTYPE in ["float16", "bfloat16", "float32"] for AT in [False, True] for BT in [False, True]
        ],
        # n-stage
        *[
            [
                (16, 16, 16, 1, 1, STAGES, 32, 32, 80, AT, BT, DTYPE, DTYPE, True),
                (64, 32, 64, 1, 2, STAGES, 128, 64, 128, AT, BT, DTYPE, DTYPE, True),
                (128, 64, 16, 1, 4, STAGES, 256, 128, 80, AT, BT, DTYPE, DTYPE, True),
                (256, 128, 32, 1, 8, STAGES, 512, 256, 160, AT, BT, DTYPE, DTYPE, True),
                (128, 128, 32, 1, 4, STAGES, 256, 256, 160, AT, BT, DTYPE, DTYPE, True),
            ] for DTYPE in ["float16", "bfloat16", "float32"] for AT in [False, True] for BT in [False, True] for STAGES in [4]
        ],
        # mixed-precision
        *[
            [
                (32, 32, 32, 1, 1, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, True),
                (128, 256, 32, 1, 8, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, True),
                (32, 64, 32, 1, 1, 2, 64, 128, 32, AT, BT, ADTYPE, BDTYPE, True),
            ] for ADTYPE, BDTYPE in [("float8e4nv", "float8e5"),
                                     ("float8e4nv", "float8e4nv"),
                                     ("float8e5", "float8e4nv"),
                                     ("float8e5", "float8e5"),
                                     ("float8e4nv", "float16"),
                                     ("float16", "float8e5"),
                                     ("float16", "float32"),
                                     ("float32", "float16"),
                                     ("bfloat16", "float32"),
                                     ("float32", "bfloat16")] for AT in [False, True] for BT in [False, True]
        ],
        # mixed-precision block layout
        *[
            [
                (32, 32, 32, 1, 1, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, False),
                (128, 256, 32, 1, 8, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, False),
                (32, 64, 32, 1, 1, 2, 64, 128, 32, AT, BT, ADTYPE, BDTYPE, False),
            ] for ADTYPE, BDTYPE in [("float8e4nv", "float16"),
                                     ("float16", "float8e5"),
                                     ("float16", "float32"),
                                     ("float32", "float16"),
                                     ("bfloat16", "float32"),
                                     ("float32", "bfloat16")] for AT in [False, True] for BT in [False, True]
        ],
        *[
            # float8e4b15 only supports row-col layout
            [
                (128, 128, 32, 1, 4, 2, None, None, None, False, True, ADTYPE, BDTYPE, True),
            ] for ADTYPE, BDTYPE in [("float8e4b15", "float8e5"),
                                     ("float8e4b15", "float16"),
                                     ("float16", "float8e4b15"),
                                     ("float8e5", "float8e5"),
                                     ("float8e4nv", "float8e4nv"),
                                     ("int8", "int8")]
        ]
    ),
)
def test_op(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, ALLOW_TF32):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        pytest.skip("Only test tl.dot() on devices with sm >= 70")
    if capability[0] < 8 and (ADTYPE == "bfloat16" or BDTYPE == "bfloat16"):
        pytest.skip("Only test bfloat16 on devices with sm >= 80")
    if capability[0] < 9 and (ADTYPE == "float8e4nv" or BDTYPE == "float8e4nv"):
        pytest.skip("Only test float8e4nv on devices with sm >= 90")
    if (ADTYPE == "bfloat16" or BDTYPE == "bfloat16") and SPLIT_K != 1:
        pytest.skip("bfloat16 matmuls don't allow split_k for now")
    torch.manual_seed(0)
    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K, 'SPLIT_K': SPLIT_K}
    pre_hook = None if SPLIT_K == 1 else lambda nargs: nargs['C'].zero_()
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE, pre_hook=pre_hook)]
    kernel = triton.ops._matmul.kernel
    kernel.configs = configs
    # kernel.run = kernel.run.run.run

    # get matrix shape
    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K * SPLIT_K if K is None else K

    def maybe_upcast(x, dtype, is_float8):
        if is_float8:
            return f8_to_f16(x, dtype)
        return x

    def init_input(m, n, dtype):
        if 'float8' in dtype:
            ewidth = {'float8e4b15': 4, 'float8e4nv': 4, 'float8e5': 5}[dtype]
            sign = torch.randint(2, size=(m, n), device="cuda", dtype=torch.int8) * 128
            val = torch.randint(2**3 - 1, size=(m, n), device="cuda", dtype=torch.int8) << 7 - ewidth
            return sign | val
        if dtype == "int8":
            return torch.randint(-128, 127, (m, n), device="cuda", dtype=torch.int8)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        exponents = torch.randint(-10, 0, size=(m, n))
        ret = (2. ** exponents).to(dtype).to("cuda")
        return ret

    # allocate/transpose inputs
    a = init_input(M, K, ADTYPE)
    b = init_input(K, N, BDTYPE)
    a = a if not AT else a.T.contiguous().T
    b = b if not BT else b.T.contiguous().T
    # run test
    a_fp8 = "float8" in ADTYPE
    b_fp8 = "float8" in BDTYPE
    th_a = maybe_upcast(a, ADTYPE, a_fp8)
    if AT and a_fp8:
        th_a = th_a.view(th_a.shape[::-1]).T
    th_b = maybe_upcast(b, BDTYPE, b_fp8)
    if BT and b_fp8:
        th_b = th_b.view(th_b.shape[::-1]).T
    if th_a.is_floating_point():
        ab_dtype = th_a.dtype if th_a.element_size() > th_b.element_size() else th_b.dtype
    else:
        ab_dtype = torch.float32
    th_c = torch.matmul(th_a.to(ab_dtype), th_b.to(ab_dtype))
    if ADTYPE == "int8" or BDTYPE == "int8":
        th_c = th_c.to(torch.int8)
    try:
        if a_fp8:
            a = triton.reinterpret(a, getattr(tl, ADTYPE))
        if b_fp8:
            b = triton.reinterpret(b, getattr(tl, BDTYPE))
        tt_c = triton.ops.matmul(a, b, None, ALLOW_TF32)
        torch.testing.assert_allclose(th_c, tt_c, atol=0, rtol=0)
    except triton.OutOfResources as e:
        pytest.skip(str(e))
