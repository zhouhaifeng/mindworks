#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
import statistics
from time import perf_counter

import numpy as np
from openvino.runtime import Core, get_version, AsyncInferQueue
from openvino.runtime.utils.types import get_dtype


def fill_tensor_random(tensor):
    dtype = get_dtype(tensor.element_type)
    rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if 0 == tensor.get_size():
        raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
    tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info('OpenVINO:')
    log.info(f"{'Build ':.<39} {get_version()}")
    if len(sys.argv) != 2:
        log.info(f'Usage: {sys.argv[0]} <path_to_model>')
        return 1
    # Optimize for throughput. Best throughput can be reached by
    # running multiple openvino.runtime.InferRequest instances asyncronously
    tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}

    # Create Core and use it to compile a model.
    # Pick a device by replacing CPU, for example MULTI:CPU(4),GPU(8).
    # It is possible to set CUMULATIVE_THROUGHPUT as PERFORMANCE_HINT for AUTO device
    core = Core()
    compiled_model = core.compile_model(sys.argv[1], 'CPU', tput)
    # AsyncInferQueue creates optimal number of InferRequest instances
    ireqs = AsyncInferQueue(compiled_model)
    # Fill input data for ireqs
    for ireq in ireqs:
        for model_input in compiled_model.inputs:
            fill_tensor_random(ireq.get_tensor(model_input))
    # Warm up
    for _ in ireqs:
        ireqs.start_async()
    ireqs.wait_all()
    # Benchmark for seconds_to_run seconds and at least niter iterations
    seconds_to_run = 10
    niter = 10
    latencies = []
    in_fly = set()
    start = perf_counter()
    time_point_to_finish = start + seconds_to_run
    while perf_counter() < time_point_to_finish or len(latencies) + len(in_fly) < niter:
        idle_id = ireqs.get_idle_request_id()
        if idle_id in in_fly:
            latencies.append(ireqs[idle_id].latency)
        else:
            in_fly.add(idle_id)
        ireqs.start_async()
    ireqs.wait_all()
    duration = perf_counter() - start
    for infer_request_id in in_fly:
        latencies.append(ireqs[infer_request_id].latency)
    # Report results
    fps = len(latencies) / duration
    log.info(f'Count:          {len(latencies)} iterations')
    log.info(f'Duration:       {duration * 1e3:.2f} ms')
    log.info('Latency:')
    log.info(f'    Median:     {statistics.median(latencies):.2f} ms')
    log.info(f'    Average:    {sum(latencies) / len(latencies):.2f} ms')
    log.info(f'    Min:        {min(latencies):.2f} ms')
    log.info(f'    Max:        {max(latencies):.2f} ms')
    log.info(f'Throughput: {fps:.2f} FPS')


if __name__ == '__main__':
    main()
