# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
import numpy as np
from pathlib import Path

import openvino.runtime.opset11 as opset11
import openvino.runtime.opset10 as opset10
from openvino.runtime import Model, serialize, Core, PartialShape, Dimension

from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph
from openvino.tools.mo.utils.logger import init_logger

# required to be in global area to run MO IR Reader
init_logger('ERROR', False)


class TestOps(unittest.TestCase):
    @staticmethod
    def check_graph_can_save(model, name):
        with tempfile.TemporaryDirectory() as tmp:
            model_xml = Path(tmp) / (name + '.xml')
            model_bin = Path(tmp) / (name + '.bin')
            serialize(model, model_xml, model_bin)
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            save_restored_graph(graph, tmp, {}, name)
            # restore 2 times to validate that after save graph doesn't lose attributes etc.
            graph, _ = restore_graph_from_ir(model_xml, model_bin)
            # check that re-saved model can be read in runtime
            Core().read_model(model_xml)
            return graph

    def test_topk_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        k_val = np.int32(3)
        axis = np.int32(1)
        topk = opset11.topk(data_parameter, k_val, axis,
                            "max", "value", stable=True, name="TopK_11")
        model = Model(topk, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'topk_model')
        topk_node = graph.get_op_nodes(op="TopK")[0]
        self.assertEqual(topk_node["version"], "opset11")
        self.assertTrue(topk_node["stable"])
        self.assertEqual(topk_node["index_element_type"], np.int32)

    def test_interpolate_11(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.int32(
            [20, 48]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'interpolate_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset11")
        self.assertTrue("force_precision_in_ports" in interpolate_node)
        self.assertEqual(interpolate_node["force_precision_in_ports"], {1: 'int64'})

    def test_interpolate_11_scales(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.float32(
            [2., 2.]), "nearest", "scales", axes=np.int32([2, 3]), name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'interpolate_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset11")
        self.assertTrue("force_precision_in_ports" not in interpolate_node)

    def test_interpolate_11_no_axes(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset11.interpolate(data_parameter, np.int32(
            [6, 12, 20, 48]), "nearest", "sizes", name="Interpolate_11")
        model = Model(interpolate, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'interpolate_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset11")
        self.assertTrue("force_precision_in_ports" in interpolate_node)
        self.assertEqual(interpolate_node["force_precision_in_ports"], {1: 'int64'})

    def test_interpolate_4(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        interpolate = opset10.interpolate(data_parameter, np.int32([20, 48]), np.float32(
            [2, 2]), "nearest", "sizes", axes=np.int32([2, 3]), name="Interpolate_4")
        model = Model(interpolate, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'interpolate4_model')
        interpolate_node = graph.get_op_nodes(op="Interpolate")[0]
        self.assertEqual(interpolate_node["version"], "opset4")

    def test_unique(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        unique = opset10.unique(data_parameter, axis=np.int32(
            [2]), sorted=True, name="Unique_10")
        model = Model(unique, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'unique_model')
        unique_node = graph.get_op_nodes(op="Unique")[0]
        self.assertEqual(unique_node["version"], "opset10")
        self.assertListEqual(unique_node.out_port(
            0).data.get_shape().tolist(), [6, 12, None, 24])
        self.assertTrue(unique_node["sorted"])

    def test_is_finite(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_finite = opset10.is_finite(data_parameter, name="Is_finite_10")
        model = Model(is_finite, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'is_finite_model')
        is_finite_node = graph.get_op_nodes(op="IsFinite")[0]
        self.assertEqual(is_finite_node["version"], "opset10")

    def test_is_inf(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_inf = opset10.is_inf(data_parameter, name="Is_inf_10")
        model = Model(is_inf, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'is_inf_model')
        is_inf_node = graph.get_op_nodes(op="IsInf")[0]
        self.assertEqual(is_inf_node["version"], "opset10")

    def test_is_nan(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset10.parameter(
            data_shape, name="Data", dtype=np.float32)
        is_nan = opset10.is_nan(data_parameter, name="Is_nan_10")
        model = Model(is_nan, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'is_nan_model')
        is_nan_node = graph.get_op_nodes(op="IsNaN")[0]
        self.assertEqual(is_nan_node["version"], "opset10")

    def test_if(self):
        parameter_x = opset11.parameter([2], np.float32, "pX")
        parameter_y = opset11.parameter([2], np.float32, "pY")
        const_z = opset11.constant(4.0, dtype=np.float32)

        condition = opset11.constant(True, dtype=bool)

        # then_body
        x_t = opset11.parameter([2], np.float32, "X")
        y_t = opset11.parameter([2], np.float32, "Y")
        mmul_t = opset11.matmul(x_t, y_t, False, False)
        mul_t = opset11.multiply(y_t, x_t)
        then_body_res_1 = opset11.result(mmul_t)
        then_body_res_2 = opset11.result(mul_t)
        then_body = Model([then_body_res_1, then_body_res_2], [x_t, y_t])

        # else_body
        x_e = opset11.parameter([2], np.float32, "X")
        z_e = opset11.parameter([], np.float32, "Z")
        mul_e = opset11.multiply(x_e, z_e)
        else_body_res_1 = opset11.result(z_e)
        else_body_res_2 = opset11.result(mul_e)
        else_body = Model([else_body_res_1, else_body_res_2], [x_e, z_e])

        if_node = opset11.if_op(condition)
        if_node.set_friendly_name("If_opset8")
        if_node.set_then_body(then_body)
        if_node.set_else_body(else_body)
        if_node.set_input(parameter_x.output(0), x_t, x_e)
        if_node.set_input(parameter_y.output(0), y_t, None)
        if_node.set_input(const_z.output(0), None, z_e)
        out1 = if_node.set_output(then_body_res_1, else_body_res_1)
        out2 = if_node.set_output(then_body_res_2, else_body_res_2)

        model = Model([out1, out2], [parameter_x, parameter_y])
        graph = TestOps.check_graph_can_save(model, 'if_model')
        if_node = graph.get_op_nodes(op="If")[0]
        self.assertEqual(if_node["version"], "opset8")
        _, layer_info, _ = if_node['IE'][0]
        _, callable_attribute = layer_info[0]
        self.assertTrue(callable(callable_attribute))
        self.assertEqual(callable_attribute(if_node), "If_opset8")

    def test_strided_slice_no_begin_end_mask(self):
        data_shape = [6, 12, 10, 24]
        data_parameter = opset11.parameter(
            data_shape, name="Data", dtype=np.float32)
        strided_slice = opset11.strided_slice(data_parameter, np.int32([1, 2, 3, 4]), np.int32(
            [3, 6, 9, 12]), np.int32([1, 1, 1, 1]), begin_mask=[], end_mask=[], name="StridedSlice_10")
        model = Model(strided_slice, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'strided_slice_model')
        strided_slice_node = graph.get_op_nodes(op="StridedSlice")[0]
        self.assertEqual(strided_slice_node["version"], "opset1")

    def test_scatter_dynamic_shape(self):
        data_parameter = opset11.parameter(
            PartialShape.dynamic(Dimension(2)), name="Data", dtype=np.float32)
        shape_of = opset11.shape_of(data_parameter)
        gather = opset11.gather(shape_of, np.int32(1), 0)
        unsqueeze = opset11.unsqueeze(gather, 0)
        scatter = opset11.scatter_update(np.int64([0, 0]), np.int64([1]), unsqueeze, axis=0)
        mul = opset11.multiply(scatter, np.int64([1, 2]))
        reshape = opset11.reshape(data_parameter, mul, True)
        model = Model(reshape, [data_parameter])
        graph = TestOps.check_graph_can_save(model, 'scatter_dynamic_model')
        scatter_update_node = graph.get_op_nodes(op="ScatterUpdate")[0]
        self.assertListEqual(scatter_update_node.out_port(0).data.get_value().tolist(), [0, None])
