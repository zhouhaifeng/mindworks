# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestShape(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        inputs_data['input'] = np.random.randint(-10, 10, input_shape).astype(self.input_type)

        return inputs_data

    def create_shape_net(self, input_shape, input_type, out_type):
        self.input_type = input_type
        types_map = {
            np.float32: tf.float32,
            np.int32: tf.int32
        }
        assert input_type in types_map, "Incorrect test case"
        tf_type = types_map[input_type]
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf_type, input_shape, 'input')
            if out_type is not None:
                tf.raw_ops.Shape(input=input, out_type=out_type)
            else:
                tf.raw_ops.Shape(input=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 3], input_type=np.float32, out_type=tf.int32),
        dict(input_shape=[3, 4, 5], input_type=np.int32, out_type=tf.int64),
        dict(input_shape=[1], input_type=np.int32, out_type=None),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_shape_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_new_frontend, use_old_api):
        self._test(*self.create_shape_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
