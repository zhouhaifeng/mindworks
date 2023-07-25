# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# expand_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'


def expand_v2(name:str, x, shape:list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        out = paddle.expand(node_x, shape=shape, name='expand_v2')

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


def expand_v2_tensor(name:str, x, out_shape, use_tensor_in_list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            out_shape[0] = paddle.assign(np.array((out_shape[0],)).astype('int32'))
            out = paddle.expand(node_x, shape=out_shape, name='expand_v2')
        else:
            out_shape = np.array(out_shape).astype('int32')
            node_shape = paddle.assign(out_shape, output=None)
            out = paddle.expand(node_x, shape=node_shape, name='expand_v2')

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
    data = np.random.rand(1, 1, 6).astype(data_type)

    expand_v2("expand_v2", data, [2, 3, -1])
    expand_v2_tensor("expand_v2_tensor", data, [2, 3, -1], False)
    expand_v2_tensor("expand_v2_tensor_list", data, [2, 3, -1], True)
    expand_v2_tensor("expand_v2_tensor_list2", data, [2, 2, 2, 3, -1], True)


if __name__ == "__main__":
    main()
