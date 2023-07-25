# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from openvino.frontend import FrontEndManager, ConversionExtension, NodeContext, OpExtension
from openvino.runtime import PartialShape, Type
import openvino.runtime.opset10 as ops

from pathlib import Path
import glob
import re
import os
import math


class aten_relu(torch.nn.Module):
    def forward(self, x):
        return x, torch.nn.functional.relu(x)


def get_scripted_model(model):
    with torch.no_grad():
        model = torch.jit.script(model)
        model.eval()
        model = torch.jit.freeze(model)
        print(model.inlined_graph)  # will help debugging
        return model


def test_pytorch_fe_set_input_shape():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_partial_shape(place, PartialShape([1, 2, 3, 4]))
    om = fe.convert(im)
    assert om.get_parameters()[0].get_partial_shape(
    ) == PartialShape([1, 2, 3, 4])


def test_pytorch_fe_set_input_type():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_element_type(place, Type.f32)
    om = fe.convert(im)
    assert om.get_parameters()[0].get_element_type() == Type.f32


def test_pytorch_fe_set_input_value():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_partial_shape(place, PartialShape([1, 2, 3, 4]))
    im.set_element_type(place, Type.f32)
    im.set_tensor_value(place, np.random.randn(1, 2, 3, 4).astype(np.float32))
    om = fe.convert(im)
    assert len(om.get_parameters()) == 0


def test_conversion_extension():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, inp):
            elu = torch.nn.functional.elu(inp, alpha=0.123)
            gelu = torch.nn.functional.gelu(elu, approximate="none")
            gelu2 = torch.nn.functional.gelu(gelu, approximate="tanh")
            softmax = torch.nn.functional.softmax(gelu2, dim=-1)
            vn = torch.linalg.vector_norm(softmax, ord=math.inf, dim=None)
            return vn

    model = Model()
    decoder = TorchScriptPythonDecoder(get_scripted_model(model))

    def convert_elu(node: NodeContext):
        inp = node.get_input(0)
        alpha = node.get_input(1)
        zero = ops.constant(np.array([0], dtype=np.float32))
        greater = ops.greater(inp, zero)
        exp = ops.exp(inp)
        one = ops.constant(np.array([0], dtype=np.float32))
        sub = ops.subtract(exp, one)
        mul = ops.multiply(sub, alpha)
        select = ops.select(greater, inp, mul)
        return select.outputs()

    def convert_gelu(node: NodeContext):
        inp = node.get_input(0)
        approximate = node.get_values_from_const_input(1)
        if approximate == "none":
            f = ops.erf(ops.divide(inp, ops.constant(np.array([math.sqrt(2.0)], dtype=np.float32))))
        elif approximate == "tanh":
            f = ops.tanh(ops.multiply(ops.constant(np.array([math.sqrt(2.0 / math.pi)], dtype=np.float32)),
                                      ops.add(inp, ops.multiply(ops.constant(np.array([0.044715], dtype=np.float32)),
                                                                ops.power(inp, ops.constant(np.array([3], dtype=np.float32)))))))
        mul = ops.multiply(ops.multiply(ops.constant(np.array([0.5], dtype=np.float32)), inp),
                           ops.add(ops.constant(np.array([1], dtype=np.float32)), f))
        return mul.outputs()

    def convert_softmax(node: NodeContext):
        inp = node.get_input(0)
        dim = node.get_values_from_const_input(1, dtype=np.int32)
        dim_const = ops.constant(np.array([dim], dtype=np.int32))
        reduce_max = ops.reduce_max(inp, dim_const, True)
        sub = ops.subtract(inp, reduce_max)
        exp = ops.exp(sub)
        reduce_sum = ops.reduce_sum(exp, dim_const, True)
        div = ops.divide(exp, reduce_sum)
        return div.outputs()


    def convert_vector_norm(node: NodeContext):
        inp = node.get_input(0)
        ord = node.get_values_from_const_input(1)
        assert ord == math.inf
        dim = node.get_values_from_const_input(2)
        if dim is None:
            inp = ops.reshape(inp, ops.constant(np.array([-1])), False)
            reduce_axes = np.array([0])
        else:
            reduce_axes = np.array(dim)
        rm = ops.reduce_max(ops.abs(inp), reduce_axes, False)
        return rm.outputs()


    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe
    fe.add_extension(ConversionExtension("aten::elu", convert_elu))
    fe.add_extension(ConversionExtension("aten::gelu", convert_gelu))
    fe.add_extension(ConversionExtension("aten::softmax", convert_softmax))
    fe.add_extension(ConversionExtension("aten::linalg_vector_norm", convert_vector_norm))
    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "Constant", "Constant",
                                                                              "Constant", "Greater", "Exp",
                                                                              "Constant", "Subtract", "Constant",
                                                                              "Multiply", "Select", "Multiply",
                                                                              "Constant", "Constant", "Divide",
                                                                              "Erf", "Add", "Multiply",
                                                                              "Multiply", "Constant", "Constant",
                                                                              "Constant", "Constant", "Power",
                                                                              "Multiply", "Add", "Multiply",
                                                                              "Tanh", "Add", "Multiply",
                                                                              "Constant", "ReduceMax", "Subtract",
                                                                              "Exp", "ReduceSum", "Divide",
                                                                              "Constant", "Reshape", "Abs",
                                                                              "Constant", "ReduceMax", "Result"]


def get_builtin_extensions_path():
    base_paths = [Path(__file__).parent.parent.parent.parent]
    repo_dir = os.environ.get("REPO_DIR")
    if repo_dir:
        base_paths.append(repo_dir)

    for base_path in base_paths:
        paths = glob.glob(os.path.join(base_path, "**", "*test_builtin_extensions*"), recursive=True)
        for path in paths:
            if re.search(r"(lib)?test_builtin_extensions.?\.(dll|so)", path):
                return path
    raise RuntimeError("Unable to find test_builtin_extensions")


def test_op_extension():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    class Elu(torch.nn.Module):
        def __init__(self, alpha):
            super(Elu, self).__init__()
            self.alpha = alpha

        def forward(self, inp):
            return torch.nn.functional.elu(inp, self.alpha)

    model = Elu(alpha=0.123)
    decoder = TorchScriptPythonDecoder(get_scripted_model(model))

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe

    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "Elu", "Result"]

    fe.add_extension(get_builtin_extensions_path())
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "CustomElu", "Result"]


def test_pytorch_telemetry():
    from openvino.frontend import TelemetryExtension
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    class MockTelemetry:
        def __init__(self, stat):
            self.stat = stat

        def send_event(self, *arg, **kwargs):
            self.stat["send_event"] += 1

        def send_error(self, *arg, **kwargs):
            self.stat["send_error"] += 1

        def send_stack_trace(self, *arg, **kwargs):
            self.stat["send_stack_trace"] += 1

    def add_ext(front_end, stat):
        tel = MockTelemetry(stat)
        front_end.add_extension(TelemetryExtension("mock",
                                                   tel.send_event,
                                                   tel.send_error,
                                                   tel.send_stack_trace))

    tel_stat = {"send_event": 0, "send_error": 0, "send_stack_trace": 0}
    # Ensure that MockTelemetry object is alive and can receive events (due to callbacks hold the object)
    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    add_ext(fe, tel_stat)
    im = fe.load(decoder)
    fe.convert(im)
    assert tel_stat["send_event"] == 2
    assert tel_stat["send_error"] == 0
    assert tel_stat["send_stack_trace"] == 0
