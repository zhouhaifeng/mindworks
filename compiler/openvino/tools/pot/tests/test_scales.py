# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest
from addict import Dict

from openvino.tools.pot import DataLoader
from openvino.tools.pot.algorithms.algorithm_selector import COMPRESSION_ALGORITHMS
from openvino.tools.pot.engines.ac_engine import ACEngine
from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.graph.node_utils import get_node_inputs, get_node_input, get_node_value
from openvino.tools.pot.graph import model_utils as mu
from openvino.tools.pot.statistics.collector import StatisticsCollector
from .utils.data_helper import dump_intermediate_data, load_json


EPS = 2e-5


def get_fq_nodes_stats_algo(model, preset, bits, is_weights, clipping_value=None):
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            './data/reference_scale/test_data')

    config = _get_pytorch_accuracy_checker_config(test_dir)

    compression_config = Dict(
        {
            'name': 'MinMaxQuantization',
            'stat_subset_size': 1,
            'preset': preset,
            'target_device': 'CPU',
            'activations': {
                'bits': bits,
                'range_estimator': {
                    'max': {
                        'clipping_value': clipping_value
                    }
                }

            },
            'weights': {
                'bits': bits,
                'mode': 'symmetric' if preset == 'performance' else 'asymmetric'
            }
        })

    engine = ACEngine(config)
    compression_config.subset_indices = [0]
    algo = COMPRESSION_ALGORITHMS.get('MinMaxQuantization')(compression_config, engine)

    model = load_model(model.model_params)

    stats_collector = StatisticsCollector(engine)
    algo.register_statistics(model, stats_collector)
    stats_collector.compute_statistics(model)

    model = algo.run(model)
    out = {}
    for fq in mu.get_nodes_by_type(model, ['FakeQuantize']):
        fq_inputs = get_node_inputs(fq)
        if is_weights and fq_inputs[0].type == 'Const':
            min_weights = np.reshape(fq_inputs[1].value, (fq_inputs[1].value.shape[0]))
            max_weights = np.reshape(fq_inputs[2].value, (fq_inputs[2].value.shape[0]))
            out[fq.name] = {'low_level': min_weights, 'high_level': max_weights}
        elif not is_weights and fq_inputs[0].type != 'Const':
            if not fq_inputs[1].value.shape:
                out[fq.name] = {'low_level': fq_inputs[1].value, 'high_level': fq_inputs[2].value}
            else:
                min_act = np.reshape(fq_inputs[1].value, (fq_inputs[1].value.shape[1]))
                max_act = np.reshape(fq_inputs[2].value, (fq_inputs[2].value.shape[1]))
                out[fq.name] = {'low_level': min_act, 'high_level': max_act}
    return out


CONFIGURATIONS = [('performance', 8,
                   os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                './data/reference_scale/mobilenet-v2-pytorch_performance_activations.json'), None),
                  ('performance', 8,
                   os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                './data/reference_scale/mobilenet-v2-pytorch_clipped_activations.json'), 3.0),
                  ('accuracy', 8,
                   os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                './data/reference_scale/mobilenet-v2-pytorch_accuracy_activations.json'), None)
                  ]


@pytest.mark.parametrize(
    'preset, bits, stats_path, clipping_value', CONFIGURATIONS,
    ids=['symmetric_{}_bits_{}_clipping_value_{}'.format(m[0], m[1], m[3]) for m in CONFIGURATIONS]
)
def test_activation_scales(tmp_path, models, preset, bits, stats_path, clipping_value):

    def normalize(_list):
        norm_coef = 0
        for fq_name in _list:
            min_level, max_level = _list[fq_name]['low_level'], _list[fq_name]['high_level']
            norm_coef = max(norm_coef, max(abs(np.mean(min_level)), abs(np.mean(max_level))))

        res = {}
        for item_name in _list:
            res[item_name] = (np.mean(_list[item_name]['low_level']) / norm_coef,
                              np.mean(_list[item_name]['high_level']) / norm_coef)

        return res

    model = models.get('mobilenet-v2-pytorch', 'pytorch', tmp_path)

    ref_nodes = load_json(stats_path)
    nodes = normalize(get_fq_nodes_stats_algo(model, preset, bits, False,
                                              clipping_value=clipping_value))
    local_path = os.path.join(tmp_path, '{}.json'.format(stats_path.split("_")[-2]))
    dump_intermediate_data(local_path, nodes)

    assert len(ref_nodes) == len(nodes)
    processed_nodes = []
    for ref_name in ref_nodes:

        ref_min, ref_max = ref_nodes[ref_name]
        node_min, node_max = nodes[ref_name]

        error = max(abs(node_min - ref_min), abs(node_max - ref_max))

        if error <= EPS and ref_name not in processed_nodes:
            processed_nodes.append((
                ref_name, abs(node_min - ref_min), abs(node_max - ref_max)))

    assert len(processed_nodes) == len(ref_nodes)


def test_weights_scales(tmp_path, models):
    path_to_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   './data/reference_scale/mobilenet-v2-pytorch_weights.json')

    model = models.get('mobilenet-v2-pytorch', 'pytorch', tmp_path)
    ref_weights = load_json(path_to_weights)
    weights = get_fq_nodes_stats_algo(model, False, 8, True)
    local_path = os.path.join(tmp_path, '{}.json'.format('mv2_weights'))
    dump_intermediate_data(local_path, weights)

    for fq_name in weights:
        item_min, item_max = weights[fq_name]['low_level'], weights[fq_name]['high_level']
        if not item_min.shape:
            continue
        shape = item_min.shape[0]

        ref_min, ref_max = ref_weights[fq_name]['low_level'], ref_weights[fq_name]['high_level']
        assert_flag = False

        if np.max(np.abs(ref_min - item_min)) < EPS and \
                np.max(np.abs(ref_max - item_max)) < EPS:
            assert_flag = True

        if not assert_flag:
            print(shape)

        assert assert_flag


def make_list(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, list):
        return x
    return [x]


def calculate_statistics(model, compression_config, engine):
    algo = COMPRESSION_ALGORITHMS.get('MinMaxQuantization')(compression_config, engine)
    stats_collector = StatisticsCollector(engine)
    algo.register_statistics(model, stats_collector)
    stats_collector.compute_statistics(model)
    return algo.run(model)


def aggregate_fq_statistics(fq_list):
    model_values = {}
    for fq in sorted(fq_list, key=lambda item: item.name):
        min_levels, max_levels = tuple([get_node_value(node)
                                        for node in get_node_inputs(fq)[1:3]])
        fq_name = fq.fullname
        if get_node_input(fq, 0).type == 'Const':
            min_levels = min_levels.reshape(min_levels.shape[0])
            max_levels = max_levels.reshape(max_levels.shape[0])
        else:
            if not min_levels.shape and not max_levels.shape:
                pass
            else:
                min_levels = min_levels.reshape(min_levels.shape[1])
                max_levels = max_levels.reshape(max_levels.shape[1])

        min_levels = make_list(min_levels)
        max_levels = make_list(max_levels)
        model_values[fq_name] = {'max': max_levels, 'min': min_levels}
    return model_values


def compare_min_max_values(model_values, refs):
    for ref_name in refs:
        refs_min_levels = make_list(refs[ref_name]['min'])
        refs_max_levels = make_list(refs[ref_name]['max'])
        min_levels = model_values[ref_name]['min']
        max_levels = model_values[ref_name]['max']

        for min_level, max_level, ref_min, ref_max in zip(
                min_levels, max_levels, refs_min_levels, refs_max_levels):
            assert abs(min_level - ref_min) < EPS
            assert abs(max_level - ref_max) < EPS


CONFIGURATIONS = [
    ('mobilenet-v2-pytorch', 'pytorch', 'symmetric'),
    ('mobilenet-v2-pytorch', 'pytorch', 'asymmetric'),
    ('mobilenet-v2-pytorch', 'pytorch', 'mixed'),
    ('resnet-50-pytorch', 'pytorch', 'symmetric'),
    ('resnet-50-pytorch', 'pytorch', 'asymmetric'),
    ('resnet-50-pytorch', 'pytorch', 'mixed'),
    ('googlenet-v3', 'tf', 'symmetric'),
    ('googlenet-v3', 'tf', 'asymmetric'),
    ('googlenet-v3', 'tf', 'mixed'),
]

REFERENCES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              './data/reference_fake_quantize_conf')


@pytest.mark.parametrize(
    'model_name, model_framework, algo_mode', CONFIGURATIONS,
    ids=['{}_{}_{}'.format(m[0], m[1], m[2]) for m in CONFIGURATIONS]
)
def test_fake_quantize_configurations(tmp_path, models, model_name, model_framework, algo_mode):
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            './data/reference_scale/test_data')

    config = _get_pytorch_accuracy_checker_config(test_dir) \
        if model_framework == 'pytorch' else _get_tf_accuracy_checker_config(test_dir)

    if algo_mode == 'symmetric':
        activations_mode, weights_mode, level_low = 'symmetric', 'symmetric', -127
    elif algo_mode == 'asymmetric':
        activations_mode, weights_mode, level_low = 'asymmetric', 'asymmetric', -128
    else:
        activations_mode, weights_mode, level_low = 'asymmetric', 'symmetric', -127

    compression_config = Dict({
        'name': 'MinMaxQuantization',
        'stat_subset_size': 1,
        'preset': 'performance',
        'target_device': 'CPU',
        'activations': {
            'bits': 8,
            'mode': activations_mode
        },
        'weights': {
            'bits': 8,
            'mode': weights_mode,
            'granularity': 'perchannel',
            'level_low': level_low,
            'level_high': 127
        }
    })

    engine = ACEngine(config)
    compression_config.subset_indices = [0]
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    model_with_stats = calculate_statistics(model, compression_config, engine)
    fq_list = mu.get_nodes_by_type(model_with_stats, ['FakeQuantize'])
    model_values = aggregate_fq_statistics(fq_list)

    refs_path = os.path.join(REFERENCES_DIR, '{}_{}.json'.format(model_name, algo_mode))
    local_path = os.path.join(tmp_path, '{}.json'.format(model_name))
    ref_exists = os.path.isfile(refs_path)
    refs = load_json(refs_path) if ref_exists else {}
    local_file = open(local_path, 'w')

    if not ref_exists:
        ref_file = open(refs_path, 'w')
        json.dump(model_values, ref_file)
        return
    json.dump(model_values, local_file)

    compare_min_max_values(model_values, refs)


MATMUL_MODELS = [
    ('matmul_3_in_slice', 'pytorch',
     {'onnx::Concat_0': [5, 10], 'onnx::Concat_1': [5, 2], 'onnx::Concat_2': [5, 10]}),
    ('matmul_3_in_cat', 'pytorch',
     {'onnx::Concat_0': [5, 10], 'onnx::Concat_1': [5, 2], 'onnx::Concat_2': [5, 10]}),
    ('matmul_3_in_agnostic', 'pytorch',
     {'onnx::Concat_0': [5, 10], 'onnx::Concat_1': [5, 2], 'onnx::Transpose_2': [5, 10]}),
    ('matmul_1_in_split', 'pytorch',
     {'onnx::Squeeze_0': [1, 3, 224, 224]}),
]


class RandomDataLoader(DataLoader):
    """ Generate random float32 data """
    def __init__(self, shapes, seed=0):
        self.rng = np.random.default_rng(seed)
        data = {}
        for i, (name, shape) in enumerate(shapes.items()):
            data[name] = self.rng.uniform(-1 + 5 * i, 1 + 5 * i, shape)
        self._data = [data]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self._data[index]


@pytest.mark.parametrize(
    'model_name, model_framework, input_shapes', MATMUL_MODELS,
    ids=['{}_{}'.format(m[0], m[1]) for m in MATMUL_MODELS]
)
def test_matmul_scale_unification(tmp_path, models, model_name, model_framework, input_shapes):
    model = models.get(model_name, model_framework, tmp_path)
    model = load_model(model.model_params)

    compression_config = Dict({
        'name': 'DefaultQuantization',
        'stat_subset_size': 1,
        'target_device': 'CPU',
    })

    data_loader = RandomDataLoader(input_shapes)
    engine = SimplifiedEngine({"device": "CPU"}, data_loader=data_loader)
    model_with_stats = calculate_statistics(model, compression_config, engine)

    fq_list = mu.get_nodes_by_type(model_with_stats, ['FakeQuantize'])
    model_values = aggregate_fq_statistics(fq_list)

    refs_path = os.path.join(REFERENCES_DIR, f'{model_name}.json')
    local_path = os.path.join(tmp_path, f'{model_name}.json')
    ref_exists = os.path.isfile(refs_path)
    refs = load_json(refs_path) if ref_exists else {}
    local_file = open(local_path, 'w')

    if not ref_exists:
        ref_file = open(refs_path, 'w')
        json.dump(model_values, ref_file)
        return
    json.dump(model_values, local_file)

    compare_min_max_values(model_values, refs)


def _get_pytorch_accuracy_checker_config(path_to_dataset):
    return Dict({
        'log_dir': './logs',
        'models': [{
            'name': 'test_conf',
            'launchers': [{
                'framework': 'dlsdk',
                'device': 'CPU',
                'adapter': 'classification'}],
            'datasets': [
                {
                    'name': 'test_dataset',
                    'data_source': path_to_dataset,
                    'reader': 'pillow_imread',
                    'preprocessing':
                        [
                            {
                                'type': 'resize',
                                'size': 256,
                                'aspect_ratio_scale': 'greater',
                                'interpolation': 'BILINEAR',
                                'use_pillow': True
                            },
                            {
                                'type': 'crop',
                                'size': 224,
                                'use_pillow': True
                            },
                            {
                                'type': 'bgr_to_rgb'
                            }]
                }
            ]}]})


def _get_tf_accuracy_checker_config(path_to_dataset):
    return Dict({
        'log_dir': './logs',
        'models': [{
            'name': 'test_conf',
            'launchers': [{
                'framework': 'dlsdk',
                'device': 'CPU',
                'adapter': 'classification'}],
            'datasets': [
                {
                    'name': 'test_dataset',
                    'data_source': path_to_dataset,
                    'reader': 'pillow_imread',
                    'preprocessing':
                        [
                            {
                                'type': 'crop',
                                'central_fraction': 0.875
                            },
                            {
                                'type': 'resize',
                                'size': 299
                            }]
                }
            ]}]})
