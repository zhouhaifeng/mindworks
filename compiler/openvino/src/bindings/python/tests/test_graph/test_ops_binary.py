# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pytest

from openvino.runtime import Type
import openvino.runtime.opset8 as ov


@pytest.mark.parametrize(
    ("graph_api_helper", "expected_type"),
    [
        (ov.add, Type.f32),
        (ov.divide, Type.f32),
        (ov.multiply, Type.f32),
        (ov.subtract, Type.f32),
        (ov.minimum, Type.f32),
        (ov.maximum, Type.f32),
        (ov.mod, Type.f32),
        (ov.equal, Type.boolean),
        (ov.not_equal, Type.boolean),
        (ov.greater, Type.boolean),
        (ov.greater_equal, Type.boolean),
        (ov.less, Type.boolean),
        (ov.less_equal, Type.boolean),
    ],
)
def test_binary_op(graph_api_helper, expected_type):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)
    parameter_b = ov.parameter(shape, name="B", dtype=np.float32)

    model = graph_api_helper(parameter_a, parameter_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("graph_api_helper", "expected_type"),
    [
        (ov.add, Type.f32),
        (ov.divide, Type.f32),
        (ov.multiply, Type.f32),
        (ov.subtract, Type.f32),
        (ov.minimum, Type.f32),
        (ov.maximum, Type.f32),
        (ov.mod, Type.f32),
        (ov.equal, Type.boolean),
        (ov.not_equal, Type.boolean),
        (ov.greater, Type.boolean),
        (ov.greater_equal, Type.boolean),
        (ov.less, Type.boolean),
        (ov.less_equal, Type.boolean),
    ],
)
def test_binary_op_with_scalar(graph_api_helper, expected_type):
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = graph_api_helper(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    "graph_api_helper",
    [ov.logical_and, ov.logical_or, ov.logical_xor],
)
def test_binary_logical_op(graph_api_helper):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)
    parameter_b = ov.parameter(shape, name="B", dtype=bool)

    model = graph_api_helper(parameter_a, parameter_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == Type.boolean


@pytest.mark.parametrize(
    "graph_api_helper",
    [ov.logical_and, ov.logical_or, ov.logical_xor],
)
def test_binary_logical_op_with_scalar(graph_api_helper):
    value_b = np.array([[False, True], [False, True]], dtype=bool)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)

    model = graph_api_helper(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == Type.boolean


@pytest.mark.parametrize(
    ("operator", "expected_type"),
    [
        (operator.add, Type.f32),
        (operator.sub, Type.f32),
        (operator.mul, Type.f32),
        (operator.truediv, Type.f32),
        (operator.eq, Type.boolean),
        (operator.ne, Type.boolean),
        (operator.gt, Type.boolean),
        (operator.ge, Type.boolean),
        (operator.lt, Type.boolean),
        (operator.le, Type.boolean),
    ],
)
def test_binary_operators(operator, expected_type):
    value_b = np.array([[4, 5], [1, 7]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("operator", "expected_type"),
    [
        (operator.add, Type.f32),
        (operator.sub, Type.f32),
        (operator.mul, Type.f32),
        (operator.truediv, Type.f32),
        (operator.eq, Type.boolean),
        (operator.ne, Type.boolean),
        (operator.gt, Type.boolean),
        (operator.ge, Type.boolean),
        (operator.lt, Type.boolean),
        (operator.le, Type.boolean),
    ],
)
def test_binary_operators_with_scalar(operator, expected_type):
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


def test_multiply():
    param_a = np.arange(48, dtype=np.int32).reshape((8, 1, 6, 1))
    param_b = np.arange(35, dtype=np.int32).reshape((7, 1, 5))

    node = ov.multiply(param_a, param_b)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [8, 7, 6, 5]
    assert node.get_output_element_type(0) == Type.i32


def test_power_v1():
    param_a = np.arange(48, dtype=np.float32).reshape((8, 1, 6, 1))
    param_b = np.arange(20, dtype=np.float32).reshape((4, 1, 5))

    node = ov.power(param_a, param_b)

    assert node.get_type_name() == "Power"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [8, 4, 6, 5]
    assert node.get_output_element_type(0) == Type.f32
