# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# leaky_relu paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def leaky_relu(name: str, x, alpha: float = 0.02, data_type='float32'):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        out = paddle.fluid.layers.leaky_relu(node_x, alpha=alpha, name='leaky_relu')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_type = 'float32'
    data = np.array([-1, 2, 3]).astype(data_type)

    leaky_relu("leaky_relu", data, 0.03)


if __name__ == "__main__":
    main()
