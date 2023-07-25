# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
import numpy as np
import pytest


@pytest.fixture()
def _proposal_node():
    attributes = {
        "base_size": np.uint16(1),
        "pre_nms_topn": np.uint16(20),
        "post_nms_topn": np.uint16(64),
        "nms_thresh": np.float64(0.34),
        "feat_stride": np.uint16(16),
        "min_size": np.uint16(32),
        "ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=np.float64),
        "scale": np.array([2, 3, 3, 4], dtype=np.float64),
    }
    batch_size = 7

    class_probs = ng.parameter([batch_size, 12, 34, 62], np.float64, "class_probs")
    bbox_deltas = ng.parameter([batch_size, 24, 34, 62], np.float64, "bbox_deltas")
    image_shape = ng.parameter([3], np.float64, "image_shape")
    return ng.proposal(class_probs, bbox_deltas, image_shape, attributes)


def test_dynamic_attributes_softmax():
    axis = 2
    data = ng.parameter([1, 2, 3, 4], np.float32, "data_in")
    node = ng.softmax(data, axis)

    assert node.get_axis() == axis
    node.set_axis(3)
    assert node.get_axis() == 3


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.uint32, np.float16),
        (np.uint32, np.float64),
    ],
)
def test_dynamic_set_attribute_value(int_dtype, fp_dtype):
    attributes = {
        "base_size": int_dtype(1),
        "pre_nms_topn": int_dtype(20),
        "post_nms_topn": int_dtype(64),
        "nms_thresh": fp_dtype(0.34),
        "feat_stride": int_dtype(16),
        "min_size": int_dtype(32),
        "ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=fp_dtype),
        "scale": np.array([2, 3, 3, 4], dtype=fp_dtype),
    }
    batch_size = 7

    class_probs = ng.parameter([batch_size, 12, 34, 62], fp_dtype, "class_probs")
    bbox_deltas = ng.parameter([batch_size, 24, 34, 62], fp_dtype, "bbox_deltas")
    image_shape = ng.parameter([3], fp_dtype, "image_shape")
    node = ng.proposal(class_probs, bbox_deltas, image_shape, attributes)

    node.set_base_size(int_dtype(15))
    node.set_pre_nms_topn(int_dtype(7))
    node.set_post_nms_topn(int_dtype(33))
    node.set_nms_thresh(fp_dtype(1.55))
    node.set_feat_stride(int_dtype(8))
    node.set_min_size(int_dtype(123))
    node.set_ratio(np.array([1.1, 2.5, 3.0, 4.5], dtype=fp_dtype))
    node.set_scale(np.array([2.1, 3.2, 3.3, 4.4], dtype=fp_dtype))
    node.set_clip_before_nms(True)
    node.set_clip_after_nms(True)
    node.set_normalize(True)
    node.set_box_size_scale(fp_dtype(1.34))
    node.set_box_coordinate_scale(fp_dtype(0.88))
    node.set_framework("OpenVINO")

    assert node.get_base_size() == int_dtype(15)
    assert node.get_pre_nms_topn() == int_dtype(7)
    assert node.get_post_nms_topn() == int_dtype(33)
    assert np.isclose(node.get_nms_thresh(), fp_dtype(1.55))
    assert node.get_feat_stride() == int_dtype(8)
    assert node.get_min_size() == int_dtype(123)
    assert np.allclose(node.get_ratio(), np.array([1.1, 2.5, 3.0, 4.5], dtype=fp_dtype))
    assert np.allclose(node.get_scale(), np.array([2.1, 3.2, 3.3, 4.4], dtype=fp_dtype))
    assert node.get_clip_before_nms()
    assert node.get_clip_after_nms()
    assert node.get_normalize()
    assert np.isclose(node.get_box_size_scale(), fp_dtype(1.34))
    assert np.isclose(node.get_box_coordinate_scale(), fp_dtype(0.88))
    assert node.get_framework() == "OpenVINO"


def test_dynamic_attr_cache(_proposal_node):
    node = _proposal_node

    assert not node._attr_cache_valid
    node.set_nms_thresh(1.3453678102)
    assert not node._attr_cache_valid
    assert np.isclose(node.get_nms_thresh(), np.float64(1.3453678102))
    assert node._attr_cache_valid


def test_dynamic_attr_transitivity(_proposal_node):
    node = _proposal_node
    node2 = node

    node.set_ratio(np.array([1.1, 2.5, 3.0, 4.5], dtype=np.float64))
    assert np.allclose(node.get_ratio(), np.array([1.1, 2.5, 3.0, 4.5], dtype=np.float64))
    assert np.allclose(node2.get_ratio(), np.array([1.1, 2.5, 3.0, 4.5], dtype=np.float64))

    node2.set_scale(np.array([2.1, 3.2, 3.3, 4.4], dtype=np.float64))
    assert np.allclose(node2.get_scale(), np.array([2.1, 3.2, 3.3, 4.4], dtype=np.float64))
    assert np.allclose(node.get_scale(), np.array([2.1, 3.2, 3.3, 4.4], dtype=np.float64))


def test_dynamic_attributes_simple():
    batch_size = 1
    input_size = 16
    hidden_size = 128

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    W_shape = [3 * hidden_size, input_size]
    R_shape = [3 * hidden_size, hidden_size]
    B_shape = [4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=np.float32)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=np.float32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=np.float32)
    parameter_R = ng.parameter(R_shape, name="R", dtype=np.float32)
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    activations = ["tanh", "relu"]
    activations_alpha = [1.0, 2.0]
    activations_beta = [1.0, 2.0]
    clip = 0.5
    linear_before_reset = True

    node = ng.gru_cell(
        parameter_X,
        parameter_H_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activations_alpha,
        activations_beta,
        clip,
        linear_before_reset,
    )

    assert node.get_hidden_size() == hidden_size
    assert all(map(lambda x, y: x == y, node.get_activations(), activations))
    assert all(np.equal(node.get_activations_alpha(), activations_alpha))
    assert all(np.equal(node.get_activations_beta(), activations_beta))
    assert node.get_linear_before_reset() == linear_before_reset
    assert np.isclose(node.get_clip(), clip)
