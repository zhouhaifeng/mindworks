# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
import numpy as np
import pytest

np_types = [np.float32, np.int32]
integral_np_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_detection_output(int_dtype, fp_dtype):
    attributes = {
        "keep_top_k": np.array([64], dtype=int_dtype),
        "nms_threshold": fp_dtype(0.645),
    }

    box_logits = ng.parameter([4, 8], fp_dtype, "box_logits")
    class_preds = ng.parameter([4, 170], fp_dtype, "class_preds")
    proposals = ng.parameter([4, 2, 10], fp_dtype, "proposals")
    aux_class_preds = ng.parameter([4, 4], fp_dtype, "aux_class_preds")
    aux_box_preds = ng.parameter([4, 8], fp_dtype, "aux_box_preds")

    node = ng.detection_output(box_logits, class_preds, proposals, attributes, aux_class_preds, aux_box_preds)

    assert node.get_type_name() == "DetectionOutput"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 1, 256, 7]


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_dynamic_get_attribute_value(int_dtype, fp_dtype):
    attributes = {
        "background_label_id": int_dtype(13),
        "top_k": int_dtype(16),
        "variance_encoded_in_target": True,
        "keep_top_k": np.array([64, 32, 16, 8], dtype=int_dtype),
        "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",
        "share_location": False,
        "nms_threshold": fp_dtype(0.645),
        "confidence_threshold": fp_dtype(0.111),
        "clip_after_nms": True,
        "clip_before_nms": False,
        "decrease_label_id": True,
        "normalized": True,
        "input_height": int_dtype(86),
        "input_width": int_dtype(79),
        "objectness_score": fp_dtype(0.77),
    }

    box_logits = ng.parameter([4, 680], fp_dtype, "box_logits")
    class_preds = ng.parameter([4, 170], fp_dtype, "class_preds")
    proposals = ng.parameter([4, 1, 8], fp_dtype, "proposals")
    aux_class_preds = ng.parameter([4, 4], fp_dtype, "aux_class_preds")
    aux_box_preds = ng.parameter([4, 680], fp_dtype, "aux_box_preds")

    node = ng.detection_output(box_logits, class_preds, proposals, attributes, aux_class_preds, aux_box_preds)

    assert node.get_background_label_id() == int_dtype(13)
    assert node.get_top_k() == int_dtype(16)
    assert node.get_variance_encoded_in_target()
    assert np.all(np.equal(node.get_keep_top_k(), np.array([64, 32, 16, 8], dtype=int_dtype)))
    assert node.get_code_type() == "caffe.PriorBoxParameter.CENTER_SIZE"
    assert not node.get_share_location()
    assert np.isclose(node.get_nms_threshold(), fp_dtype(0.645))
    assert np.isclose(node.get_confidence_threshold(), fp_dtype(0.111))
    assert node.get_clip_after_nms()
    assert not node.get_clip_before_nms()
    assert node.get_decrease_label_id()
    assert node.get_normalized()
    assert node.get_input_height() == int_dtype(86)
    assert node.get_input_width() == int_dtype(79)
    assert np.isclose(node.get_objectness_score(), fp_dtype(0.77))
