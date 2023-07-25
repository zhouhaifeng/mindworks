# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class quantized_add_relu(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor1, input_tensor2):
        quantized_tensor1 =  torch.quantize_per_tensor(input_tensor1, self.scale, self.zero_point, self.dtype)
        quantized_tensor2 =  torch.quantize_per_tensor(input_tensor2, self.scale, self.zero_point, self.dtype)
        quantized_add_relu = torch.ops.quantized.add_relu(quantized_tensor1, quantized_tensor2, self.scale, self.zero_point)
        dequantized_tensor = torch.dequantize(quantized_add_relu)
        return dequantized_tensor

class TestQuantizedAddReLU(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(5.00 * np.random.rand(100, 100) + 5.00, dtype=np.float32),
                np.array(5.00 * np.random.rand(100, 100) + 5.00, dtype=np.float32))

    @pytest.mark.parametrize("scale", [
        1.0, 0.21, 0.62
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8, 
        torch.qint8
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_add_relu(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(quantized_add_relu(scale, zero_point, dtype), None, ["quantized::add_relu"], 
                ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)
