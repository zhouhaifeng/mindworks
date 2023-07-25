# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np
from generator import generator, generate

from openvino.runtime import Core
from openvino.tools.ovc.convert import convert_model


@generator
class TestMoFreezePlaceholderTFFE(unittest.TestCase):
    def basic(self, input_model, argv_input, inputs, dtype, expected, only_conversion=False):
        path = os.path.dirname(__file__)
        input_model = os.path.join(path, "test_models", input_model)

        try:
            model = convert_model(input_model, input=argv_input)
        except Exception as ex:
            self.fail("Model conversion failed due to error: {}".format(ex))

        if only_conversion:
            return

        ie = Core()
        exec_net = ie.compile_model(model, "CPU")
        req = exec_net.create_infer_request()
        results = req.infer(inputs)
        values = list(results.values())[0]
        if dtype is not None:
            assert values.dtype == dtype
        assert np.allclose(values, expected)

    @generate(
        *[
            (
                    "in1[1 4]->[1.0 2.0 3.0 4.0],in2[1 4]{f32}->[1.0 2.0 3.0 4.0]",
                    {},
                    np.array([2.0, 4.0, 6.0, 8.0]),
                    np.float32,
            ),
            (
                    "in2{f32}->[0.0 0.0 0.0 0.0]",
                    {"in1": np.array([[1.0, 2.0], [3.0, 4.0]])},
                    np.array([[1.0, 2.0], [3.0, 4.0]]),
                    np.float32,
            ),
            (
                    "in2->[1.0 15.0 15.5 1.0]",
                    {"in1": np.array([[2.0, 4.0], [12.0, 8.0]])},
                    np.array([[3.0, 19.0], [27.5, 9.0]]),
                    np.float32,
            ),
            (
                    "in1[1 4]{i32}->[1 2 3 4],in2[1 4]{i32}->[1 2 3 4]",
                    {},
                    np.array([2.0, 4.0, 6.0, 8.0]),
                    np.int32,
            ),
        ],
    )
    def test_fp32(self, input_freezing_value, inputs, expected,
                  dtype):
        self.basic("model_fp32.pbtxt", input_freezing_value, inputs, dtype, expected)

    @generate(
        *[
            (
                    "in1[1 4]->[1 2 3 4],in2[1 4]{i32}->[1 2 3 4]",
                    {},
                    np.array([1, 4, 9, 16]),
                    np.int32,
            ),
            (
                    "in2->[2 5 6 7 3 2]",
                    {"in1": np.array([[2, 4, 1], [1, 2, 8]])},
                    np.array([[4, 20, 6], [7, 6, 16]]),
                    np.int32,
            ),
        ],
    )
    def test_int32(self, input_freezing_value, inputs, expected,
                   dtype=None):
        self.basic("model_int32.pbtxt", input_freezing_value, inputs, dtype, expected)

    @generate(
        *[
            (
                    "in1[2]->[True False],in2[2]->[True True]",
                    {},
                    np.array([True, False], dtype=bool),
                    bool,
            ),
            (
                    "in2[2,3]->[True,True,False,True,True,False]",
                    {"in1": np.array([[False, True, True], [False, True, True]], dtype=bool)},
                    np.array([[False, True, False], [False, True, False]], dtype=bool),
                    bool,
            ),
            (
                    "in2[]->True",
                    {"in1": np.array([[False, True, True], [False, True, True]], dtype=bool)},
                    np.array([[False, True, True], [False, True, True]], dtype=bool),
                    bool,
            ),
        ],
    )
    def test_bool(self, input_freezing_value, inputs, expected,
                  dtype=None):
        self.basic("model_bool.pbtxt", input_freezing_value, inputs, dtype, expected)

    @generate(
        *[
            (
                    "in1[3]->[1 2 3],in2[3]->[4 5 6],cond->False",
                    {},
                    np.array([4, 5, 6], dtype=np.float32),
                    np.float32,
                    None
            ),
            (
                    "in1,in2,cond->False",
                    {"in1": np.array([2.0, 4.0, 6.0], dtype=np.float32),
                     "in2": np.array([1.0, 3.0, 5.0], dtype=np.float32)},
                    np.array([2, 4, 6], dtype=np.float32),
                    np.float32,
                    True  # fill a bug to investigate why compilation of this model is hang on
            ),
            # case: input_shape + freeze_placeholder_with_value
            (
                    "in2,in1->[2.0 4.0 6.0],cond->True",
                    {"in2": np.array([1.0, 3.0, 5.0], dtype=np.float32)},
                    np.array([2, 4, 6], dtype=np.float32),
                    np.float32,
                    False
            ),
        ],
    )
    def test_bool2(self, input_freezing_value, inputs, expected,
                   dtype=None, only_conversion=False):
        self.basic("model_bool2.pbtxt", input_freezing_value, inputs, dtype, expected,
                   only_conversion)

    @generate(
        *[
            (
                    "add:0[3],z",
                    {"add:0": np.array([4, 5, 6], dtype=np.float32), "z": np.array([1, 2, 3], dtype=np.float32)},
                    np.array([4, 10, 18], dtype=np.float32),
                    np.float32,
                    None
            ),
            (
                    "add:0{i32}[3],z{i32}",
                    {"add:0": np.array([4, 5, 6], dtype=np.int32), "z": np.array([1, 2, 3], dtype=np.int32)},
                    np.array([4, 10, 18], dtype=np.int32),
                    np.int32,
                    None
            ),
        ],
    )
    def test_cutting_fp32(self, input_freezing_value, inputs, expected,
                          dtype=None, only_conversion=False):
        self.basic("model_three_inputs.pbtxt", input_freezing_value, inputs, dtype, expected,
                   only_conversion)

    @generate(
        *[
            (
                    "x[1,4],y[4]",
                    {"x": np.array([[3, 2, 1, 5]], dtype=np.int32), "y": np.array([0, -1, -7, 8], dtype=np.int32)},
                    np.array([[3, 1, -6, 13]], dtype=np.int32),
                    np.int32,
                    None
            ),
            (
                    "x,y",
                    {"x": np.array([[-3, 20, 1]], dtype=np.int32), "y": np.array([[10, -11, -17]], dtype=np.int32)},
                    np.array([[7, 9, -16]], dtype=np.int32),
                    np.int32,
                    None
            ),
            (
                    "x",
                    {"x": np.array([[-3, 20, 1]], dtype=np.int32)},
                    np.array([[-2, 22, 4], [1, 25, 7]], dtype=np.int32),
                    np.int32,
                    None
            ),
        ],
    )
    def test_placeholder_with_default(self, inputs, inputs_data, expected,
                                      dtype=None,  only_conversion=False):
        self.basic("placeholder_with_default.pbtxt", inputs, inputs_data, dtype, expected,
                   only_conversion)

    @generate(
        *[
            (
                    "x[4],y->2.0",
                    {"x": np.array([3, 2, 1, 5], dtype=np.float32)},
                    np.array([6, 4, 2, 10], dtype=np.float32),
                    np.float32,
                    None
            ),
            (
                    "x[1],y->[2.0,3.0]",
                    {"x": np.array([3], dtype=np.float32)},
                    np.array([6, 9], dtype=np.float32),
                    np.float32,
                    None
            ),
        ],
    )
    def test_freeze_placeholder_with_unknown_rank(self, inputs, inputs_data, expected,
                                                  dtype=None, only_conversion=False):
        self.basic("mul_with_unknown_rank_y.pbtxt", inputs, inputs_data, dtype, expected,
                   only_conversion)

    def test_conversion_model_oneshot_iterator_default(self):
        self.basic("model_oneshot_iterator.pbtxt", None, None, None, None, True)

    @generate(
        *[
            (
                    "in2{f32}->[0.0 0.0 0.0 0.0]",
                    {"in1": np.array([[1.0, 2.0], [3.0, 4.0]])},
                    np.array([[1.0, 2.0], [3.0, 4.0]]),
                    np.float32,
            ),
            (
                    "in2->[1.0 15.0 15.5 1.0]",
                    {"in1": np.array([[2.0, 4.0], [12.0, 8.0]])},
                    np.array([[3.0, 19.0], [27.5, 9.0]]),
                    np.float32,
            ),
        ],
    )
    @unittest.skip("109220: Use generating script for this test model instead of Git LFS")
    def test_conversion_model_with_non_standard_extension(self, input_freezing_value, inputs, expected,
                                                          dtype):
        self.basic("model_fp32.frozen", input_freezing_value, inputs, dtype, expected, only_conversion=False)

    @unittest.skip("109220: Make TF FE to return the error")
    def test_conversion_dir_model(self):
        with self.assertRaisesRegex(Exception,
                                    "Internal error or inconsistent input model: the frontend supports "
                                    "only frozen binary protobuf format."):
            self.basic(".", None, None, None, None,
                       only_conversion=True)

    @generate(
        *[
            (
                    {"x": np.array([1, 2], dtype=np.int32), "y": np.array([4], dtype=np.int32)},
                    np.array([-3, -2], dtype=np.int32),
                    np.int32,
            ),
            (
                    {"x": np.array([20, 25], dtype=np.int32), "y": np.array([10], dtype=np.int32)},
                    np.array([30, 35], dtype=np.int32),
                    np.int32,
            )
        ],
    )
    def test_conversion_pbtxt_model_with_inference(self, inputs, expected, dtype):
        self.basic("model_with_if.pbtxt", None, inputs, dtype, expected, only_conversion=False)

    @generate(
        *[
            # new frontend
            (
                    "model_add_with_undefined_constant.pbtxt",
                    "x[2,3]",
                    {"x": np.array([[12, 13, 10], [11, 14, 16]], dtype=np.float32)},
                    np.array([[12, 13, 10], [11, 14, 16]], dtype=np.float32),
                    np.float32
            ),
            (
                    "model_mul_with_undefined_constant.pbtxt",
                    "x[2]",
                    {"x": np.array([11, -12], dtype=np.int32)},
                    np.array([0, 0], dtype=np.int32),
                    np.int32
            ),
        ],
    )
    def test_conversion_model_with_undefined_constant(self, model_name, argv_input, inputs, expected, dtype):
        self.basic(model_name, argv_input, inputs, dtype, expected, only_conversion=False)
