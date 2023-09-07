// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_iouring_thread.h"

using namespace std;

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const long long int num_bytes,
                           const bool validate)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename(filename),
      _num_bytes(num_bytes),
      _validate(validate)
{
    _cpu_buffer = _buffer.is_cuda() ? _buffer.to(torch::kCPU).pin_memory() : _buffer;
    _contiguous_buffer = _cpu_buffer.contiguous();
}

char* io_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void io_op_desc_t::fini()
{
    if (_read_op && _buffer.is_cuda()) { _buffer.copy_(_cpu_buffer.to(torch::kCUDA)); }
}

deepspeed_iouring_thread_t::deepspeed_iouring_thread_t(const int tid, deepspeed_iouring_config_t& iouring_config)
    : _tid(tid),
      _iouring_config(iouring_config),
      _iouring_ctxt(new iouring_context(iouring_config._block_size, iouring_config._queue_depth)),
      _time_to_exit(false)
{
}

deepspeed_iouring_thread_t::~deepspeed_iouring_thread_t() {}

void deepspeed_iouring_thread_t::run()
{
    while (true) {
        std::shared_ptr<struct io_op_desc_t> next_io_op = nullptr;

        {
            std::unique_lock<std::mutex> lock(_work_sync._mutex);
            _work_sync._cond_var.wait(lock,
                                      [this] { return (!_work_queue.empty() || _time_to_exit); });
            if (!_work_queue.empty()) {
                next_io_op = _work_queue.front();
                _work_queue.pop();
            }
        }

        if (next_io_op) {
            const auto base_offset = next_io_op->_num_bytes * _tid;

            std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
                next_io_op->_fd, base_offset, next_io_op->_num_bytes, next_io_op->data_ptr()));

            if (_iouring_config._overlap_events) {
                do_iouring_operation_overlap(
                    next_io_op->_read_op, _iouring_ctxt, xfer_ctxt, &_iouring_config, nullptr);
            } else {
                do_iouring_operation_sequential(
                    next_io_op->_read_op, _iouring_ctxt, xfer_ctxt, &_iouring_config, nullptr);
            }

            {
                std::lock_guard<std::mutex> lock(_complete_sync._mutex);
                _complete_queue.push(next_io_op);
            }
            _complete_sync._cond_var.notify_one();
        }

        if (_time_to_exit) { break; }
    }
}
