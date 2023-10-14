# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.runtime.swap_tensor.constants import *

UIO_DEFAULT_DICT = {
    UIO_BLOCK_SIZE: UIO_BLOCK_SIZE_DEFAULT,
    UIO_QUEUE_DEPTH: UIO_QUEUE_DEPTH_DEFAULT,
    UIO_THREAD_COUNT: UIO_THREAD_COUNT_DEFAULT,
    UIO_SINGLE_SUBMIT: UIO_SINGLE_SUBMIT_DEFAULT,
    UIO_OVERLAP_EVENTS: UIO_OVERLAP_EVENTS_DEFAULT
}


def get_uio_config(param_dict):
    if UIO in param_dict.keys() and param_dict[UIO] is not None:
        uio_dict = param_dict[UIO]
        return {
            UIO_BLOCK_SIZE: get_scalar_param(uio_dict, UIO_BLOCK_SIZE, UIO_BLOCK_SIZE_DEFAULT),
            UIO_QUEUE_DEPTH: get_scalar_param(uio_dict, UIO_QUEUE_DEPTH, UIO_QUEUE_DEPTH_DEFAULT),
            UIO_THREAD_COUNT: get_scalar_param(uio_dict, UIO_THREAD_COUNT, UIO_THREAD_COUNT_DEFAULT),
            UIO_SINGLE_SUBMIT: get_scalar_param(uio_dict, UIO_SINGLE_SUBMIT, UIO_SINGLE_SUBMIT_DEFAULT),
            UIO_OVERLAP_EVENTS: get_scalar_param(uio_dict, UIO_OVERLAP_EVENTS, UIO_OVERLAP_EVENTS_DEFAULT)
        }

    return UIO_DEFAULT_DICT
