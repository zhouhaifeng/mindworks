# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestNarrow(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.dim, self.start, self.length)

    def create_model(self):

        class aten_narrow(torch.nn.Module):

            def forward(self, input_tensor, dim: int, start, length: int):
                return torch.narrow(input_tensor, dim=dim, start=start, length=length)

        ref_net = None

        return aten_narrow(), ref_net, "aten::narrow"

    @pytest.mark.parametrize("input_tensor", [
        np.random.randn(3, 3), np.random.randn(3, 4, 5)
    ])
    @pytest.mark.parametrize("dim", [
        np.array(0).astype(np.int32), np.array(1).astype(np.int32), np.array(-1).astype(np.int32)
    ])
    @pytest.mark.parametrize("start", [
        np.array(0).astype(np.int32), np.array(1).astype(np.int32)
    ])
    @pytest.mark.parametrize("length", [
        np.array(1).astype(np.int32), np.array(2).astype(np.int32)
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_narrow(self, input_tensor, dim, start, length, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        self.dim = dim
        self.start = start
        self.length = length
        self._test(*self.create_model(), ie_device, precision, ir_version)