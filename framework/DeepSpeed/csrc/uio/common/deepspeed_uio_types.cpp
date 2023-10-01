// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <cmath>
#include "liburing.h"
#include "deepspeed_uio_utils.h"

using namespace std;

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

deepspeed_uio_config_t::deepspeed_uio_config_t()
    : _block_size(c_block_size),
      _queue_depth(c_io_queue_depth),
      _single_submit(false),
      _overlap_events(false),
      _lock_memory(false)
{
}

deepspeed_uio_config_t::deepspeed_uio_config_t(int entries = 64, uint32_t flags = 0, uint32_t wq_fd = 0)
    : _block_size(block_size),
      _queue_depth(queue_depth),
      _single_submit(single_submit),
      _overlap_events(overlap_events),
      _lock_memory(lock_memory)
{
    io_uring_params p = {
        .flags = flags,
        .wq_fd = wq_fd,
    };

    io_uring_queue_init_params(entries, &ring, &p);
}

void deepspeed_uio_latency_t::dump(const std::string tag)
{
    std::cout << tag << _min_usec << " " << _max_usec << " " << _avg_usec << " " << std::endl;
}

void deepspeed_uio_latency_t::accumulate(const struct deepspeed_uio_latency_t& other)
{
    _min_usec += other._min_usec;
    _max_usec += other._max_usec;
    _avg_usec += other._avg_usec;
}

void deepspeed_uio_latency_t::scale(const float scaler)
{
    _min_usec *= scaler;
    _max_usec *= scaler;
    _avg_usec *= scaler;
}
