# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc
import tensorflow as tf
import numpy as np

class TestBiasAdd(CommonTFLayerTest):
    def create_bias_add_placeholder_const_net(self, shape, ir_version, use_new_frontend, output_type=tf.float32):
        """
            Tensorflow net                      IR net

            Placeholder->BiasAdd       =>       Placeholder->Add
                         /                                   /
            Const-------/                       Const-------/

        """

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)
            tf_y_shape = tf_x_shape[-1:]

            x = tf.compat.v1.placeholder(output_type, tf_x_shape, 'Input')
            constant_value = np.random.randint(0, 1, tf_y_shape).astype(output_type.as_numpy_dtype())
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            tf.nn.bias_add(x, y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    def create_bias_add_2_consts_net(self, shape, ir_version, use_new_frontend, output_type=tf.float32):
        """
            Tensorflow net                         IR net

            Const->BiasAdd-->Concat       =>       Const---->Concat
                    /        /                                  /
            Const--/        /                      Placeholder-/
                           /
            Placeholder---/

        """

        #
        #   Create Tensorflow model
        #

        tf.compat.v1.reset_default_graph()

        tf_concat_axis = -1

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)
            tf_y_shape = tf_x_shape[-1:]

            constant_value_x = np.random.randint(-256, 256, tf_x_shape).astype(output_type.as_numpy_dtype())
            x = tf.constant(constant_value_x)
            constant_value_y = np.random.randint(-256, 256, tf_y_shape).astype(output_type.as_numpy_dtype())
            y = tf.constant(constant_value_y)

            add = tf.nn.bias_add(x, y, name="Operation")

            placeholder = tf.compat.v1.placeholder(output_type, tf_x_shape,
                                                   'Input')  # Input_1 in graph_def

            concat = tf.concat([placeholder, add], axis=tf_concat_axis, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_2D = [
        dict(shape=[1, 1]),
        dict(shape=[1, 224])
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_bias_add_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_3D = [
        pytest.param(dict(shape=[1, 1, 224]), marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(shape=[1, 3, 224]), marks=pytest.mark.xfail(reason="*-19053"))
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_bias_add_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [
        dict(shape=[1, 1, 100, 224]),
        pytest.param(dict(shape=[1, 3, 100, 224]), marks=pytest.mark.precommit_tf_fe),
        pytest.param(dict(shape=[1, 3, 100, 224], output_type=tf.float16), marks=pytest.mark.precommit_tf_fe)
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bias_add_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_5D = [
        dict(shape=[1, 1, 50, 100, 224]),
        dict(shape=[1, 3, 220, 222, 224])
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bias_add_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
