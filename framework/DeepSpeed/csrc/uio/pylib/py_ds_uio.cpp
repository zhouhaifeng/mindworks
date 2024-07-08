// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <torch/extension.h>
#include "deepspeed_py_uio_handle.h"
#include "deepspeed_py_copy.h"

#include "liburing/io_service.hpp"
#include "liburing/sqe_awaitable.hpp"
#include "liburing/task.hpp"
#include "liburing/utils.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("uio_read", &deepspeed_py_uio_read, "DeepSpeed Asynchronous I/O Read");

    m.def("uio_write", &deepspeed_py_uio_write, "DeepSpeed Asynchronous I/O Write");

    m.def("deepspeed_memcpy", &deepspeed_py_memcpy, "DeepSpeed Memory Copy");

    //todo: remove followling    
    py::class_<deepspeed_uio_handle_t>(m, "uio_handle")
        .def(py::init<const int, const int, const bool, const bool, const int>())
        .def("get_single_submit", &deepspeed_uio_handle_t::get_single_submit)
        .def("get_thread_count", &deepspeed_uio_handle_t::get_thread_count)

        .def("read", &deepspeed_uio_handle_t::read)
        .def("write", &deepspeed_uio_handle_t::write)

        .def("pread", &deepspeed_uio_handle_t::pread)
        .def("pwrite", &deepspeed_uio_handle_t::pwrite)

        .def("sync_pread", &deepspeed_uio_handle_t::sync_pread)
        .def("sync_pwrite", &deepspeed_uio_handle_t::sync_pwrite)
        .def("async_pread", &deepspeed_uio_handle_t::async_pread)
        .def("async_pwrite", &deepspeed_uio_handle_t::async_pwrite)

        .def("new_cpu_locked_tensor", &deepspeed_uio_handle_t::new_cpu_locked_tensor)
        .def("free_cpu_locked_tensor", &deepspeed_uio_handle_t::free_cpu_locked_tensor)

        .def("wait", &deepspeed_uio_handle_t::wait);
}
