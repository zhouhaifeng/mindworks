# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestRepeat(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 10).astype(np.float32),)

    def create_model(self, repeats):
        import torch

        class aten_repeat(torch.nn.Module):
            def __init__(self, repeats):
                super(aten_repeat, self).__init__()
                self.repeats = repeats

            def forward(self, x):
                return x.repeat(self.repeats)

        ref_net = None

        return aten_repeat(repeats), ref_net, "aten::repeat"

    @pytest.mark.parametrize("repeats", [(4, 3), (1, 1), (1, 2, 3), (1, 2, 2, 3)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_repeat(self, repeats, ie_device, precision, ir_version):
        self._test(*self.create_model(repeats), ie_device, precision, ir_version)

class TestRepeatList(PytorchLayerTest):
    def _prepare_input(self, repeats_shape):
        import numpy as np
        return (np.random.randn(2, 10).astype(np.float32), np.random.randn(*repeats_shape).astype(np.float32),)

    def create_model(self):
        import torch

        class aten_repeat(torch.nn.Module):

            def forward(self, x, y):
                y_shape = y.shape
                return x.repeat([y_shape[0], y_shape[1]])

        ref_net = None

        return aten_repeat(), ref_net, ["aten::repeat", "prim::ListConstruct"]

    @pytest.mark.parametrize("repeats", [(4, 3), (1, 1), (1, 3, 3), (1, 2, 2, 3)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_repeat(self, repeats, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,  kwargs_to_prepare_input={"repeats_shape": repeats})
