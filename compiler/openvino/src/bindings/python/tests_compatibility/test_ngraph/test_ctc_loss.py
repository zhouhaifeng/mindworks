# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from ngraph.impl import Type


def test_ctc_loss_props():
    ind_dtype = np.int32
    float_dtype = np.float32
    logits = ng.parameter([2, 100, 80], dtype=float_dtype, name="logits")
    logit_length = ng.parameter([2], dtype=ind_dtype, name="logit_length")
    labels = ng.parameter([2, 100], dtype=ind_dtype, name="labels")
    label_length = ng.parameter([2], dtype=ind_dtype, name="label_length")
    blank_index = ng.parameter([], dtype=ind_dtype, name="blank_index")
    preprocess_collapse_repeated = False
    ctc_merge_repeated = True
    unique = False

    node = ng.ctc_loss(logits, logit_length, labels, label_length, blank_index,
                       preprocess_collapse_repeated, ctc_merge_repeated, unique)
    assert node.get_type_name() == "CTCLoss"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2]
    assert node.get_output_element_type(0) == Type.f32
