# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest

d2_params = [{'kernel_size': [3, 3], 'stride': 1, 'padding': 0},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': 1},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [0, 1]},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [1, 0]},
             {'kernel_size': [3, 3], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': None, 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [], 'padding': 0},
             ]

d1_params = [{'kernel_size': 3, 'stride': 1, 'padding': 0},
             {'kernel_size': (4,), 'stride': 1, 'padding': 1},
             {'kernel_size': 4, 'stride': (5,), 'padding': 2},
             {'kernel_size': 4, 'stride': None, 'padding': 0},
             ]
d3_params = [{'kernel_size': [3, 3, 3], 'stride': 1, 'padding': 0},
             {'kernel_size': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 1},
             {'kernel_size': [3, 3, 3], 'stride': [
                 3, 3, 3], 'padding': [0, 0, 0]},
             {'kernel_size': [3, 2, 1], 'stride': [
                 3, 1, 1], 'padding': [0, 0, 0]},
             {'kernel_size': [3, 2, 1], 'stride': None, 'padding': [0, 0, 0]},
             ]


class TestPooling(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        import numpy as np
        shape = (1, 3, 15, 15, 15)
        return (np.random.randn(*shape[:ndim]).astype(np.float32),)

    def create_model(self, op_type, kernel_size, stride, padding, dilation=1, ceil_mode=True, count_include_pad=True):
        import torch

        class aten_avg_pooling_base(torch.nn.Module):
            def __init__(self):
                super(aten_avg_pooling_base, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.ceil_mode = ceil_mode
                self.count_include_pad = count_include_pad

            def forward(self, x):
                pass

        class aten_max_pooling_base(torch.nn.Module):
            def __init__(self):
                super(aten_max_pooling_base, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.ceil_mode = ceil_mode

            def forward(self, x):
                pass

        class aten_avg_pool2d(aten_avg_pooling_base):
            def forward(self, x):
                return torch.nn.functional.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)

        class aten_avg_pool3d(aten_avg_pooling_base):
            def forward(self, x):
                return torch.nn.functional.avg_pool3d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)

        class aten_avg_pool1d(aten_avg_pooling_base):
            def forward(self, x):
                return torch.nn.functional.avg_pool1d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)

        class aten_max_pool2d(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode)

        class aten_max_pool3d(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode)

        class aten_max_pool1d(aten_max_pooling_base):
            def forward(self, x):
                return torch.nn.functional.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode)

        ops = {
            "max_pool1d": aten_max_pool1d,
            "max_pool2d": aten_max_pool2d,
            "max_pool3d": aten_max_pool3d,
            "avg_pool1d": aten_avg_pool1d,
            "avg_pool2d": aten_avg_pool2d,
            "avg_pool3d": aten_avg_pool3d
        }

        ref_net = None
        aten_pooling = ops[op_type]

        return aten_pooling(), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_avg_pool1d(self, params, ceil_mode, count_include_pad, ie_device, precision, ir_version):
        self._test(*self.create_model("avg_pool1d", **params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={'ndim': 3}, trace_model=True,
                   dynamic_shapes=False)

    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_avg_pool2d(self, params, ceil_mode, count_include_pad, ie_device, precision, ir_version):
        self._test(*self.create_model("avg_pool2d", **params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, trace_model=True, dynamic_shapes=False)

    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_avg_pool3d(self, params, ceil_mode, count_include_pad, ie_device, precision, ir_version):
        self._test(*self.create_model("avg_pool3d", **params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={'ndim': 5}, trace_model=True,
                   dynamic_shapes=False)

    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_max_pool1d(self, params, ceil_mode, dilation, ie_device, precision, ir_version):
        self._test(*self.create_model("max_pool1d", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={'ndim': 3}, dynamic_shapes=False)

    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_max_pool2d(self, params, ceil_mode, dilation, ie_device, precision, ir_version):
        to_trace = False
        if params["stride"] == []:
            to_trace = True
        self._test(*self.create_model("max_pool2d", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version, dynamic_shapes=False, trace_model=to_trace)

    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_max_pool3d(self, params, ceil_mode, dilation, ie_device, precision, ir_version):
        self._test(*self.create_model("max_pool3d", **params, ceil_mode=ceil_mode, dilation=dilation),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={'ndim': 5}, dynamic_shapes=False)
