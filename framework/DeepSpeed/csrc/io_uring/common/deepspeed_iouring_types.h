// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <libaio.h>
#include <stdlib.h>

#include <string>
#include <vector>

using namespace std;

struct deepspeed_iouring_latency_t {
    double _min_usec;
    double _max_usec;
    double _avg_usec;

    void dump(const std::string tag);
    void accumulate(const deepspeed_iouring_latency_t&);
    void scale(const float value);
};

struct deepspeed_iouring_perf_t {
    deepspeed_iouring_latency_t _submit;
    deepspeed_iouring_latency_t _complete;
    double _e2e_usec;
    double _e2e_rate_GB;
};

struct deepspeed_iouring_config_t {
    struct io_uring ring;
    const int _block_size;
    const int _queue_depth;
    const bool _single_submit;
    const bool _overlap_events;
    const bool _lock_memory;

    deepspeed_iouring_config_t();
    deepspeed_iouring_config_t(const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const bool lock_memory);
};

/*
io_uring不需要context
struct iouring_context {
    std::vector<struct io_event> _io_events;
    std::vector<struct iocb*> _iocbs;
    int _block_size;
    int _queue_depth;

    iouring_context(const int block_size, const int queue_depth);
    ~iouring_context();
};
*/