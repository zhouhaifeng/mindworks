# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, undefined_shape_of_rank
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.broadcast import Broadcast
from unit_tests.utils.graph import build_graph, valued_const_with_data, regular_op_with_empty_data, \
    shaped_data


@generator
class BroadcastTest(unittest.TestCase):
    @generate(*[
        ([1], [3, 3], None, 'numpy', [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([1], [3, 3], None, 'numpy'),

        # shape broadcasting
        ([1], [1, 2], [0], 'explicit'),
        ([1], [1, 2], [-2], 'explicit'),
        ([1, 7], [5, 1, 7, 3], [1, 2], 'explicit'),
        ([2, 1, 3], [2, 1, 3, 3], [0, 1, 2], 'explicit'),
        ([2, 1, 3], [5, 2, 1, 3], [1, 2, 3], 'explicit'),

        # value broadcasting
        ([1], [1, 2], [0], 'explicit', [[1, 1]]),

        ([[3, 1]], [2, 1, 2], [1, 2], 'explicit', [[[3, 1]], [[3, 1]]]),  # ref_shape (2, 1, 2)

        ([[3, 1]], [2, 1, 2], [-2, -1], 'explicit', [[[3, 1]], [[3, 1]]]),  # ref_shape (2, 1, 2)

        ([[[9, 5, 7]], [[9, 5, 7]]], [2, 2, 1, 3], [1, 2, 3], 'explicit',  # in_shape (2, 1, 3)
         [[[[9, 5, 7]], [[9, 5, 7]]], [[[9, 5, 7]], [[9, 5, 7]]]]),  # ref_out_shape (2, 2, 1, 3)

        ([[[9, 5, 7]], [[3, 4, 8]]], [2, 1, 3, 3], [0, 1, 2], 'explicit',  # in_shape (2, 1, 3)
         [[[[9, 9, 9], [5, 5, 5], [7, 7, 7]]], [[[3, 3, 3], [4, 4, 4], [8, 8, 8]]]]),  # ref_out_shape (2, 1, 3, 3)

        # negative tests
        ([1], [2, 2], [0], 'explicit', None, True),
        ([1, 7], [5, 2, 7, 3], [1, 2], 'explicit', None, True),
        ([1, 7], [5, 2, 7, 3], [2, 1], 'explicit', None, True),
        ([1, 7], [5, 2, 7, 3], [-3, -2], 'explicit', None, True),
    ])
    def test_broadcast(self, data, target_shape, axes_mapping=None, mode='numpy', ref_out=None, test_raising=False):
        if ref_out is not None:
            input = valued_const_with_data('data', int64_array(data))
        else:
            input = shaped_data('data', int64_array(data))

        nodes = {
            **input,
            **valued_const_with_data('target_shape', int64_array(target_shape)),
            **regular_op_with_empty_data('broadcast', {'op': 'Broadcast', 'mode': mode}),
        }

        edges = [('data', 'broadcast'),
                 ('target_shape', 'broadcast'),
                 ('broadcast', 'broadcast_d')]

        if axes_mapping is not None:
            nodes.update(**valued_const_with_data('axes_mapping', int64_array(axes_mapping)))
            edges.append(('axes_mapping', 'broadcast'))
        graph = build_graph(nodes, edges)

        broadcast_node = Node(graph, 'broadcast')
        if test_raising:
            self.assertRaises(AssertionError, Broadcast.infer, broadcast_node)
            return

        Broadcast.infer(broadcast_node)
        if ref_out is not None:
            self.assertTrue(np.array_equal(broadcast_node.out_node().value, np.array(ref_out)))
        else:
            self.assertTrue(np.array_equal(broadcast_node.out_node().shape, np.array(target_shape)))

    @generate(*[
        ([1], [3], [0], 'explicit', undefined_shape_of_rank(3)),
        ([1], [3], None, 'numpy', undefined_shape_of_rank(3)),
        ([1], [3], None, 'bidirectional', undefined_shape_of_rank(3)),
        ([1, 7], [4], [1, 2], 'explicit', undefined_shape_of_rank(4)),
        ([1, 2], [3], None, 'numpy', undefined_shape_of_rank(3)),
        ([1, 1], [2], None, 'bidirectional', undefined_shape_of_rank(2)),
        ([1, 1], [2, 1], None, 'numpy', None, True),
    ])
    def test_broadcast_dynamic(self, data, target_shape_shape, axes_mapping=None, mode='numpy', ref_out_shape=None, test_raising=False):
        nodes = {
            **shaped_data('data', int64_array(data)),
            **shaped_data('target_shape', int64_array(target_shape_shape)),
            **regular_op_with_empty_data('broadcast', {'op': 'Broadcast', 'mode': mode}),
        }

        edges = [('data', 'broadcast'),
                 ('target_shape', 'broadcast'),
                 ('broadcast', 'broadcast_d')]

        if axes_mapping is not None:
            nodes.update(**valued_const_with_data('axes_mapping', int64_array(axes_mapping)))
            edges.append(('axes_mapping', 'axes_mapping_d'))
            edges.append(('axes_mapping_d', 'broadcast'))
        graph = build_graph(nodes, edges)

        broadcast_node = Node(graph, 'broadcast')
        if test_raising:
            self.assertRaises(AssertionError, Broadcast.infer, broadcast_node)
            return

        Broadcast.infer(broadcast_node)
        self.assertTrue(np.array_equal(broadcast_node.out_node().shape, ref_out_shape))
