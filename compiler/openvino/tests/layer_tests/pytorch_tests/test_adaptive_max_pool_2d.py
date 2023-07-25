# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestAdaptiveMaxPool2D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, output_size=None, return_indices=False):
        class aten_adaptive_max_pool2d(torch.nn.Module):

            def __init__(self, output_size=None, return_indices=False) -> None:
                super().__init__()
                self.output_size = output_size
                self.return_indices = return_indices

            def forward(self, input_tensor):
                if self.return_indices:
                    output, indices = F.adaptive_max_pool2d(input_tensor, self.output_size, True)
                    return output
                return F.adaptive_max_pool2d(input_tensor, self.output_size, False)

        ref_net = None

        return aten_adaptive_max_pool2d(output_size, return_indices), ref_net, "aten::adaptive_max_pool2d"

    @pytest.mark.parametrize('input_tensor', ([
        np.random.randn(1, 1, 4, 4).astype(np.float32),
        np.random.randn(1, 3, 32, 32).astype(np.float32)
    ]))
    @pytest.mark.parametrize('output_size', ([
        [2, 2],
        [4, 4],
    ]))
    @pytest.mark.parametrize('return_indices', ([
        False,
        True,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_ts_backend
    def test_adaptive_max_pool2d(self, ie_device, precision, ir_version, input_tensor, output_size, return_indices):
        self.input_tensor = input_tensor
        self._test(*self.create_model(output_size, return_indices), ie_device, precision, ir_version)
