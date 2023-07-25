# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestClamp(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, minimum, maximum, as_tensors=False):
        import torch

        class aten_clamp(torch.nn.Module):
            def __init__(self, minimum, maximum, as_tensors):
                super(aten_clamp, self).__init__()
                if minimum is not None and as_tensors:
                    minimum = torch.tensor(minimum)
                self.min = minimum
                if maximum is not None and as_tensors:
                    maximum = torch.tensor(maximum)
                self.max = maximum

            def forward(self, x):
                return torch.clamp(x, self.min, self.max)

        ref_net = None
        op_name = "aten::clamp"
        return aten_clamp(minimum, maximum, as_tensors), ref_net, op_name

    @pytest.mark.parametrize("minimum,maximum",
                             [(0., 1.), (-0.5, 1.5), (None, 10.), (None, -10.), (10., None), (-10., None), (100, 200)])
    @pytest.mark.parametrize("as_tensors", [True, False])
    @pytest.mark.nightly
    def test_clamp(self, minimum, maximum, as_tensors, ie_device, precision, ir_version):
        self._test(*self.create_model(minimum, maximum, as_tensors), ie_device, precision, ir_version)

    @pytest.mark.xfail(reason='OpenVINO clamp does not support min > max')
    def test_clamp_min_greater(self, ie_device, precision, ir_version):
        self._test(*self.create_model(1.0, 0.0), ie_device, precision, ir_version)


class TestClampMin(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, minimum, as_tensor=False):
        import torch

        class aten_clamp_min(torch.nn.Module):
            def __init__(self, minimum, as_tensor):
                super(aten_clamp_min, self).__init__()
                self.min = torch.tensor(minimum) if as_tensor else minimum

            def forward(self, x):
                return torch.clamp_min(x, self.min)

        ref_net = None
        op_name = "aten::clamp_min"
        return aten_clamp_min(minimum, as_tensor), ref_net, op_name

    @pytest.mark.parametrize("minimum", [0., 1., -1., 0.5])
    @pytest.mark.parametrize("as_tensor", [True, False])
    @pytest.mark.nightly
    def test_clamp_min(self, minimum, as_tensor, ie_device, precision, ir_version):
        self._test(*self.create_model(minimum, as_tensor), ie_device, precision, ir_version)


class TestClampMax(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, maximum, as_tensor=False):
        import torch

        class aten_clamp_max(torch.nn.Module):
            def __init__(self, maximum, as_tensor):
                super(aten_clamp_max, self).__init__()
                self.max = torch.tensor(maximum) if as_tensor else maximum

            def forward(self, x):
                return torch.clamp_max(x, self.max)

        ref_net = None
        op_name = "aten::clamp_max"
        return aten_clamp_max(maximum, as_tensor), ref_net, op_name

    @pytest.mark.parametrize("maximum", [0., 1., -1., 0.5])
    @pytest.mark.parametrize("as_tensor", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_clamp(self, maximum, as_tensor, ie_device, precision, ir_version):
        self._test(*self.create_model(maximum, as_tensor), ie_device, precision, ir_version)
