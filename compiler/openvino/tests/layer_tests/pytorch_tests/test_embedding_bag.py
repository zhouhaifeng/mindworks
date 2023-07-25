# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestEmbeddingBag1dOffsets(PytorchLayerTest):
    def _prepare_input(self, indicies_dtype, per_sample_weights=False):
        import numpy as np
        indices = np.array([2, 2, 2, 2, 4, 3, 2, 9]).astype(indicies_dtype)
        weights = np.random.randn(10, 10).astype(np.float32)
        offsets = np.array([0, 4]).astype(indicies_dtype)
        if per_sample_weights:
            per_sample_weights = np.random.randn(
                *indices.shape).astype(np.float32)
            return (indices, weights, offsets, per_sample_weights)
        return (indices, weights, offsets)

    def create_model(self, per_sample_weights):
        import torch
        import torch.nn.functional as F

        class aten_embedding_bag(torch.nn.Module):
            def __init__(self, per_sample_weights=False) -> None:
                super().__init__()
                if per_sample_weights:
                    self.forward = self.forward_offsets_per_sample_weights

            def forward(self, indicies, weight, offsets):
                return F.embedding_bag(indicies, weight, offsets, mode="sum")

            def forward_offsets_per_sample_weights(self, indicies, weight, offsets, per_sample_wights):
                return F.embedding_bag(indicies, weight, offsets, mode="sum", per_sample_weights=per_sample_wights)

        ref_net = None

        return aten_embedding_bag(per_sample_weights), ref_net, "aten::embedding_bag"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("indicies_dtype", ["int", "int32"])
    @pytest.mark.parametrize("per_sample_weights", [True, False])
    def test_embedding_bag(self, ie_device, precision, ir_version, indicies_dtype, per_sample_weights):
        self._test(*self.create_model(per_sample_weights), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"indicies_dtype": indicies_dtype, "per_sample_weights": per_sample_weights}, 
                   trace_model=True, dynamic_shapes=not per_sample_weights)


class TestEmbeddingBag2d(PytorchLayerTest):
    def _prepare_input(self, indicies_size, indicies_dtype, per_sample_weights):
        import numpy as np
        indices = np.random.randint(
            0, 9, size=indicies_size).astype(indicies_dtype)
        weights = np.random.randn(10, 10).astype(np.float32)
        if per_sample_weights:
            per_sample_weights = np.random.randn(
                *indices.shape).astype(np.float32)
            return (indices, weights, per_sample_weights)
        return (indices, weights)

    def create_model(self, per_sample_weights):
        import torch
        import torch.nn.functional as F

        class aten_embedding_bag(torch.nn.Module):
            def __init__(self, per_sample_weights=False) -> None:
                super().__init__()
                if per_sample_weights:
                    self.forward = self.forward_per_sample_weights

            def forward(self, indicies, weight):
                return F.embedding_bag(indicies, weight, mode="sum")

            def forward_per_sample_weights(self, indicies, weight, per_sample_wights):
                return F.embedding_bag(indicies, weight, mode="sum", per_sample_weights=per_sample_wights)

        ref_net = None

        return aten_embedding_bag(per_sample_weights), ref_net, "aten::embedding_bag"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("indicies_size", [[1, 1], [2, 5], [3, 10], [4, 7]])
    @pytest.mark.parametrize("indicies_dtype", ["int", "int32"])
    @pytest.mark.parametrize("per_sample_weights", [True, False])
    def test_embedding_bag(self, ie_device, precision, ir_version, indicies_dtype, indicies_size, per_sample_weights):
        self._test(*self.create_model(per_sample_weights), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"indicies_size": indicies_size, "indicies_dtype": indicies_dtype, "per_sample_weights": per_sample_weights}, 
                   trace_model=True, dynamic_shapes=not per_sample_weights)
