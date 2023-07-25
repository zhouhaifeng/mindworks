# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_tensor', (np.random.randn(1, 2, 8, 9, 10).astype(np.float32),
                                          np.random.randn(2, 8, 9, 10).astype(np.float32)))
@pytest.mark.parametrize('output_size', ([5, 7, 9], 7))
class TestAdaptiveAvgPool3D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, output_size):
        class aten_adaptive_avg_pool3d(torch.nn.Module):

            def __init__(self, output_size) -> None:
                super().__init__()
                self.output_size = output_size

            def forward(self, input_tensor):
                return torch.nn.functional.adaptive_avg_pool3d(input_tensor, self.output_size)

        ref_net = None

        return aten_adaptive_avg_pool3d(output_size), ref_net, "aten::adaptive_avg_pool3d"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_ts_backend
    def test_adaptive_avg_pool3d(self, ie_device, precision, ir_version, input_tensor, output_size):
        self.input_tensor = input_tensor
        self._test(*self.create_model(output_size), ie_device, precision, ir_version)
