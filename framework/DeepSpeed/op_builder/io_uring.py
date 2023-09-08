# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import distutils.spawn
import subprocess

from .builder import OpBuilder


class IOUringBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_IOURING"
    NAME = "io_uring"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.iouring.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/io_uring/py_lib/deepspeed_py_copy.cpp', 'csrc/aio/py_lib/py_ds_iouring.cpp',
            'csrc/io_uring/py_lib/deepspeed_py_iouring.cpp', 'csrc/aio/py_lib/deepspeed_py_iouring_handle.cpp',
            'csrc/io_uring/py_lib/deepspeed_iouring_thread.cpp', 'csrc/io_uring/common/deepspeed_iouring_utils.cpp',
            'csrc/io_uring/common/deepspeed_iouring_common.cpp', 'csrc/io_uring/common/deepspeed_iouring_types.cpp',
            'csrc/io_uring/py_lib/deepspeed_pin_tensor.cpp'
        ]

    def include_paths(self):
        return ['csrc/io_uring/py_lib', 'csrc/io_uring/common']

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return [
            '-g',
            '-Wall',
            '-O0',
            '-std=c++14',
            '-shared',
            '-fPIC',
            '-Wno-reorder',
            CPU_ARCH,
            '-fopenmp',
            SIMD_WIDTH,
            '-laio',
        ]

    def extra_ldflags(self):
        return ['-luring']

    def check_for_libaio_pkg(self):
        libs = dict(
            dpkg=["-l", "liburing-dev", "apt"],
            pacman=["-Q", "liburing", "pacman"],
            rpm=["-q", "liburing-devel", "yum"],
        )

        found = False
        for pkgmgr, data in libs.items():
            flag, lib, tool = data
            path = distutils.spawn.find_executable(pkgmgr)
            if path is not None:
                cmd = f"{pkgmgr} {flag} {lib}"
                result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                if result.wait() == 0:
                    found = True
                else:
                    self.warning(f"{self.NAME}: please install the {lib} package with {tool}")
                break
        return found

    def is_compatible(self, verbose=True):
        # Check for the existence of liburing by using distutils
        # to compile and link a test program that calls io_submit,
        # which is a function provided by liburing that is used in the async_io op.
        # If needed, one can define -I and -L entries in CFLAGS and LDFLAGS
        # respectively to specify the directories for liburing.h and liburing.so.
        iouring_compatible = self.has_function('io_uring_submit', ('aio', ))
        if verbose and not iouring_compatible:
            self.warning(f"{self.NAME} requires the dev liburing .so object and headers but these were not found.")

            # Check for the liburing package via known package managers
            # to print suggestions on which package to install.
            self.check_for_liburing_pkg()

            self.warning(
                "If liburing is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found."
            )
        return super().is_compatible(verbose) and iouring_compatible
