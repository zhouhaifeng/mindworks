# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestFakeQuantizePerTensorAffine(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(3, 2, 2).astype(np.float32),)

    def create_model(self, scale, zero_point, quant_min, quant_max):
        class fake_quantize_per_tensor_affine(torch.nn.Module):
            def __init__(self, scale, zero_point, quant_min, quant_max):
                super(fake_quantize_per_tensor_affine, self).__init__()
                self.scale = scale
                self.zero_point = zero_point
                self.quant_min = quant_min
                self.quant_max = quant_max

            def forward(self, x):
                return torch.fake_quantize_per_tensor_affine(
                    x, self.scale, self.zero_point, self.quant_min, self.quant_max
                )

        ref_net = None

        return (
            fake_quantize_per_tensor_affine(scale, zero_point, quant_min, quant_max),
            ref_net,
            "aten::fake_quantize_per_tensor_affine",
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "scale, zero_point, quant_min, quant_max",
        [
            (1.0, 1, 0, 255),
            (0.01, 0, 0, 255),
            (-0.01, 0, 0, 255),
            (0.5, 0, -128, 127),
            (0.5, -1, -128, 127),
            (1.0, 0, 0, 127),
        ],
    )
    def test_fake_quantize_per_tensor_affine(
        self, ie_device, precision, ir_version, scale, zero_point, quant_min, quant_max
    ):
        self._test(
            *self.create_model(scale, zero_point, quant_min, quant_max),
            ie_device,
            precision,
            ir_version,
            freeze_model=False
        )


class TestFakeQuantizePerChannelAffine(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(3, 2, 2).astype(np.float32),)

    def create_model(self, scale, zero_point, axis, quant_min, quant_max):
        class fake_quantize_per_channel_affine(torch.nn.Module):
            def __init__(self, scale, zero_point, axis, quant_min, quant_max):
                super(fake_quantize_per_channel_affine, self).__init__()
                self.scale = scale
                self.zero_point = zero_point
                self.axis = axis
                self.quant_min = quant_min
                self.quant_max = quant_max

            def forward(self, x):
                return torch.fake_quantize_per_channel_affine(
                    x, self.scale, self.zero_point, self.axis, self.quant_min, self.quant_max
                )

        ref_net = None

        return (
            fake_quantize_per_channel_affine(scale, zero_point, axis, quant_min, quant_max),
            ref_net,
            "aten::fake_quantize_per_channel_affine",
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "scale, zero_point, axis, quant_min, quant_max",
        [
            (torch.tensor([0.005, 0.7]), torch.zeros(2), 1, 0, 255),
            (torch.tensor([1.5, -0.7, -0.1]), torch.tensor([1, 0, -1], dtype=torch.int32), 0, -128, 127),
            (torch.tensor([-0.005, 0.7]), torch.tensor([0, 1], dtype=torch.int32), 1, 0, 127),
            (torch.tensor([-0.005, -0.7, 0.1]), torch.tensor([1, 0, 1], dtype=torch.int32), 0, 0, 255),
        ],
    )
    def test_fake_quantize_per_channel_affine(
        self, ie_device, precision, ir_version, scale, zero_point, axis, quant_min, quant_max
    ):
        self._test(
            *self.create_model(scale, zero_point, axis, quant_min, quant_max),
            ie_device,
            precision,
            ir_version,
            freeze_model=False
        )
