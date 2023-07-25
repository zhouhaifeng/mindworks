# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSqueeze(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        inputs_data['input'] = np.random.randint(-50, 50, input_shape).astype(self.input_type)

        return inputs_data

    def create_squeeze_net(self, input_shape, axis, input_type=np.float32):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Squeeze(input=input, axis=axis)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1], axis=[0], input_type=np.float32),
        dict(input_shape=[3, 1], axis=[], input_type=np.int32),
        dict(input_shape=[2, 3, 1], axis=[-1], input_type=np.float32),
        dict(input_shape=[1, 10, 1, 5], axis=[0, 2], input_type=np.float32),
        dict(input_shape=[1, 22, 1, 1, 10], axis=[0, 2, -2], input_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_squeeze_basic(self, params, ie_device, precision, ir_version, temp_dir,
                           use_new_frontend, use_old_api):
        self._test(*self.create_squeeze_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_1D = [
        dict(input_shape=[1], axis=[], input_type=np.float32),
        dict(input_shape=[1], axis=[-1], input_type=np.float32)
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_squeeze_1D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_2D = [
        dict(input_shape=[1, 2], axis=[0], input_type=np.float32),
        dict(input_shape=[4, 1], axis=[-1], input_type=np.int32)
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_squeeze_2D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_3D = [
        dict(input_shape=[2, 1, 3], axis=[], input_type=np.float32),
        dict(input_shape=[1, 2, 3], axis=[0], input_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_squeeze_3D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [
        dict(input_shape=[1, 1, 5, 10], axis=[], input_type=np.int32),
        dict(input_shape=[1, 1, 5, 10], axis=[0], input_type=np.float32),
        dict(input_shape=[3, 1, 5, 1], axis=[-1], input_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_squeeze_4D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_5D = [
        dict(input_shape=[1, 1, 5, 10, 22], axis=[], input_type=np.float32),
        dict(input_shape=[1, 1, 5, 10, 22], axis=[0], input_type=np.int32),
        dict(input_shape=[22, 1, 5, 10, 1], axis=[-1], input_type=np.float32),
        dict(input_shape=[1, 22, 1, 1, 10], axis=[0, 3], input_type=np.int32),
        dict(input_shape=[1, 1, 10, 1, 1], axis=[0, 1, 3], input_type=np.float32),
        dict(input_shape=[1, 1, 1, 1, 22], axis=[0, 1, 2, 3], input_type=np.float32)
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_squeeze_5D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
