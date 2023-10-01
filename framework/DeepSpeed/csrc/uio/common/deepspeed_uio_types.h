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

struct deepspeed_uio_latency_t {
    double _min_usec;
    double _max_usec;
    double _avg_usec;

    void dump(const std::string tag);
    void accumulate(const deepspeed_uio_latency_t&);
    void scale(const float value);
};

struct deepspeed_uio_perf_t {
    deepspeed_uio_latency_t _submit;
    deepspeed_uio_latency_t _complete;
    double _e2e_usec;
    double _e2e_rate_GB;
};

struct deepspeed_uio_config_t {
    struct io_uring ring;

    deepspeed_uio_config_t();
    deepspeed_uio_config_t(int entries = 64, uint32_t flags = 0, uint32_t wq_fd = 0);
};


//struct uio_context {
//    struct io_uring ring;
//
//    uio_context(const int block_size, const int queue_depth);
//    ~uio_context();
//};
