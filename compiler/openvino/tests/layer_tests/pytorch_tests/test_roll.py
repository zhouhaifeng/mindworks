# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestRoll(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, (2, 3, 4)).astype(np.float32),)

    def create_model(self, shifts, dim):
        import torch

        class aten_roll(torch.nn.Module):
            def __init__(self, shifts, dim=None):
                super(aten_roll, self).__init__()
                self.dim = dim
                self.shits = shifts

            def forward(self, x):
                if self.dim is not None:
                    return torch.roll(x, self.shits, self.dim)
                return torch.roll(x, self.shits)

        ref_net = None

        return aten_roll(shifts, dim), ref_net, "aten::roll"

    @pytest.mark.parametrize(("shifts", "dim"), [
        [(2, 1), (0, 1)],
        [1, 0],
        [-1, 0],
        [1, None],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_roll(self, shifts, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(shifts, dim), ie_device, precision, ir_version)
