# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ngraph as ng
from ngraph.impl import Shape, Type


def test_log_softmax():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ng.log_softmax(data, 1)
    assert node.get_type_name() == "LogSoftmax"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32
