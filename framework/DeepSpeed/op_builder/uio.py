# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import distutils.spawn
import subprocess

from .builder import OpBuilder


class UIOBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_UIO"
    NAME = "uio"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.uio.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/uio/py_lib/deepspeed_py_copy.cpp', 'csrc/uio/py_lib/py_ds_uio.cpp',
            'csrc/uio/py_lib/deepspeed_py_uio.cpp', 'csrc/uio/py_lib/deepspeed_py_uio_handle.cpp',
            'csrc/uio/py_lib/deepspeed_uio_thread.cpp', 'csrc/uio/common/deepspeed_uio_utils.cpp',
            'csrc/uio/common/deepspeed_uio_common.cpp', 'csrc/uio/common/deepspeed_uio_types.cpp',
            'csrc/uio/py_lib/deepspeed_pin_tensor.cpp'
        ]

    def include_paths(self):
        return ['csrc/uio/py_lib', 'csrc/uio/common']

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return [
            '-g',
            '-Wall',
            '-O0',
            '-std=c++20',
            '-shared',
            '-fPIC',
            '-Wno-reorder',
            CPU_ARCH,
            '-fopenmp',
            SIMD_WIDTH,
            '-luring',
            '-fcoroutines', 
            '-lcoroutine', 
            'Icsrc/includes/liburing' 
        ]

    def extra_ldflags(self):
        return ['-luring']

    #todo: 需确认csrc/includes/liburing与liburing-dev里的liburing是否冲突
    def check_for_liburing_pkg(self):
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
        uio_compatible = self.has_function('io_uring_submit', ('uio', ))
        if verbose and not uio_compatible:
            self.warning(f"{self.NAME} requires the dev liburing .so object and headers but these were not found.")

            # Check for the liburing package via known package managers
            # to print suggestions on which package to install.
            self.check_for_liburing_pkg()

            self.warning(
                "If liburing is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found."
            )
        return super().is_compatible(verbose) and uio_compatible
