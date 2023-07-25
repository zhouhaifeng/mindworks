# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from openvino.runtime import Model  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.moc_frontend.preprocessing import apply_preprocessing


def moc_emit_ir(ngraph_function: Model, argv: argparse.Namespace):

    # Apply preprocessing (mean/scale/reverse_channels/convert_layout/etc)
    apply_preprocessing(ov_function=ngraph_function, argv=argv)

    # Apply transformations
    from openvino.tools.ovc.moc_frontend.offline_transformations import apply_user_transformations, apply_moc_transformations, \
        apply_moc_legacy_transformations, apply_fused_names_cleanup

    apply_moc_transformations(ngraph_function)
    from openvino._offline_transformations import compress_quantize_weights_transformation # pylint: disable=no-name-in-module,import-error
    compress_quantize_weights_transformation(ngraph_function)

    if argv.framework == "onnx":  # TODO: Consider removing
        # set OldApi map in IR to be executed via OV API 1.x and for parity with legacy MO
        params_with_custom_types = [] if argv.placeholder_data_types is None \
            else list(argv.placeholder_data_types.keys())
        apply_moc_legacy_transformations(ngraph_function, params_with_custom_types)

    # TODO: Move compression to save_model at the level of main function where serialize is called
    if not argv.is_python_api_used and argv.compress_to_fp16:
        from openvino.tools.ovc.moc_frontend.offline_transformations import compress_model
        compress_model(ngraph_function)

    apply_fused_names_cleanup(ngraph_function)

    del argv.feManager
    return ngraph_function
