# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestWhile(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['x'] = np.random.randint(1, 10, x_shape).astype(np.int32)
        inputs_data['y'] = np.random.randint(-50, 50, y_shape).astype(np.int32)
        return inputs_data

    def create_while_net(self, y_shape, data_type, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def while_function(x, y):
            @tf.function
            def cond(x, y):
                return tf.less(x, 10)

            @tf.function
            def body(x, y):
                y_new = tf.add(y, tf.constant(2, dtype=data_type))
                x_new = tf.add(x, 1)
                return x_new, y_new

            return tf.while_loop(cond, body, [x, y])

        tf_while_graph = tf.function(while_function)
        x = np.random.randint(1, 10, []).astype(data_type)
        y = np.random.randint(-50, 50, y_shape).astype(data_type)
        concrete_func = tf_while_graph.get_concrete_function(x, y)

        # lower_control_flow defines representation of While operation
        # in case of lower_control_flow=True it is decomposed into LoopCond, NextIteration and TensorArray operations
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        return graph_def, None

    test_data_basic = [
        dict(y_shape=[2, 3], data_type=np.int32, lower_control_flow=False),
        dict(y_shape=[2, 1, 4], data_type=np.int32, lower_control_flow=False),
        pytest.param(dict(y_shape=[2, 1, 4], data_type=np.int32, lower_control_flow=True),
                     marks=pytest.mark.xfail(reason="105670"))
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_while_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_new_frontend, use_old_api):
        self._test(*self.create_while_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestWhileShapeVariant(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['x'] = np.random.randint(1, 10, x_shape).astype(np.int32)
        inputs_data['y'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_while_net(self, y_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def while_function(x, y):
            @tf.function
            def cond(x, y):
                return tf.less(x, 10)

            @tf.function
            def body(x, y):
                add_2 = tf.add(y, tf.constant(2, dtype=tf.float32))
                y_new = tf.concat(values=[y, add_2], axis=0)
                x_new = tf.add(x, tf.constant(1, tf.int32))
                return x_new, y_new

            return tf.while_loop(cond, body, [x, y],
                                 shape_invariants=[tf.TensorShape([]),
                                                   tf.TensorShape([None] + y_shape[1:])])

        tf_while_graph = tf.function(while_function)
        x = np.random.randint(1, 10, []).astype(np.int32)
        y = np.random.randint(-50, 50, y_shape).astype(np.float32)
        concrete_func = tf_while_graph.get_concrete_function(x, y)

        # lower_control_flow defines representation of While operation
        # in case of lower_control_flow=True it is decomposed into LoopCond, NextIteration and TensorArray operations
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        return graph_def, None

    test_data_basic = [
        dict(y_shape=[2, 3], lower_control_flow=False),
        dict(y_shape=[2, 1, 4], lower_control_flow=False),
        pytest.param(dict(y_shape=[2, 1, 4], lower_control_flow=True),
                     marks=pytest.mark.xfail(reason="105670"))
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_while_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_new_frontend, use_old_api):
        self._test(*self.create_while_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
