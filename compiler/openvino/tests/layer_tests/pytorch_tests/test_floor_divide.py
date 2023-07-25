# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestFloorDivide(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.other_tensor)

    def create_model(self):
        import torch

        class aten_floor_divide(torch.nn.Module):
            def __init__(self):
                super(aten_floor_divide, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.floor_divide(input_tensor, other_tensor)

        ref_net = None

        return aten_floor_divide(), ref_net, "aten::floor_divide"

    def create_model_int(self):
        import torch

        class aten_floor_divide(torch.nn.Module):
            def __init__(self):
                super(aten_floor_divide, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.floor_divide(input_tensor.to(torch.int32), other_tensor.to(torch.int64))

        ref_net = None

        return aten_floor_divide(), ref_net, "aten::floor_divide"

    @pytest.mark.parametrize('input_tensor', ([
        np.random.randn(5).astype(np.float32),
        np.random.randn(5, 5, 1).astype(np.float32),
        np.random.randn(1, 1, 5, 5).astype(np.float32),
    ]))
    @pytest.mark.parametrize('other_tensor', ([
        np.array([[0.5]]).astype(np.float32),
        np.random.randn(5).astype(np.float32),
        np.random.randn(5, 1).astype(np.float32),
        np.random.randn(1, 5).astype(np.float32),
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_floor_divide(self, input_tensor, other_tensor, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        self.other_tensor = other_tensor
        self._test(*self.create_model(), ie_device, precision, ir_version, trace_model=True)

    @pytest.mark.parametrize('input_tensor', ([
        np.random.randint(low=0, high=10, size=5).astype(np.float32),
        np.random.randint(low=1, high=10, size=(5, 5, 1)).astype(np.float32),
        np.random.randint(low=1, high=10, size=(1, 1, 5, 5)).astype(np.float32),
    ]))
    @pytest.mark.parametrize('other_tensor', ([
        np.array([[2]]).astype(np.float32),
        np.random.randint(low=1, high=10, size=5).astype(np.float32),
        np.random.randint(low=1, high=10, size=(5, 1)).astype(np.float32),
        np.random.randint(low=1, high=10, size=(1, 5)).astype(np.float32),
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_floor_divide_int(self, input_tensor, other_tensor, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        self.other_tensor = other_tensor
        self.create_model = self.create_model_int
        self._test(*self.create_model(), ie_device, precision, ir_version)
