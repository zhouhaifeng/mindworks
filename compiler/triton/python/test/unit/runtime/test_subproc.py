import multiprocessing
import os
import shutil
from collections import namedtuple

import torch

import triton
import triton.language as tl

tmpdir = ".tmp"


def reset_tmp_dir():
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)


instance_descriptor = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"])


def compile_fn(config, cc):
    @triton.jit
    def kernel_sub(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) - tl.load(b + idx) * 777)
    triton.compile(
        fn=kernel_sub,
        signature={0: "*fp32", 1: "*fp32", 2: "*fp32"},
        device=0,
        constants={3: 32},
        configs=[config],
        warm_cache_only=True,
        cc=cc,
    )


def test_compile_in_subproc() -> None:
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    config = instance_descriptor(tuple(range(4)), (), (), ())

    multiprocessing.set_start_method('fork')
    proc = multiprocessing.Process(
        target=compile_fn,
        args=(config, cc))
    proc.start()
    proc.join()
    assert proc.exitcode == 0


def compile_fn_dot(config, cc):
    @triton.jit
    def kernel_dot(Z):
        offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        z = tl.load(Z + offs)
        z = tl.dot(z, z)
        tl.store(Z + offs, z)

    triton.compile(
        fn=kernel_dot,
        signature={0: "*fp32"},
        device=0,
        configs=[config],
        warm_cache_only=True,
        cc=cc,
    )


def test_compile_in_forked_subproc() -> None:
    reset_tmp_dir()
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    config = instance_descriptor(tuple(range(1)), (), (), ())

    assert multiprocessing.get_start_method() == 'fork'
    proc = multiprocessing.Process(
        target=compile_fn_dot,
        args=(config, cc))
    proc.start()
    proc.join()
    assert proc.exitcode == 0
