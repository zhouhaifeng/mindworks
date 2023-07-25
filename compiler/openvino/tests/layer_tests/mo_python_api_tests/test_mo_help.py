# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import sys
import unittest
from openvino.tools.ovc import ovc
from openvino.tools.ovc.cli_parser import get_mo_convert_params
from openvino.tools.mo.utils.cli_parser import get_mo_convert_params as legacy_mo_params
from pathlib import Path

from common.utils.common_utils import shell


class TestSubprocessMoConvert(unittest.TestCase):
    def test_mo_convert(self):
        mo_convert_params = get_mo_convert_params()

        # Test ovc tool help
        mo_path = Path(ovc.__file__).parent
        mo_runner = mo_path.joinpath('main.py').as_posix()
        params = [sys.executable, mo_runner, "--help"]
        _, mo_output, _ = shell(params)

        # We don't expect PyTorch specific parameters to be in help message of the MO tool.
        for group in mo_convert_params:
            if group == 'Pytorch-specific parameters:' or group == 'PaddlePaddle-specific parameters:':
                continue
            for param_name in group:
                assert param_name in mo_output

        # Test Python API help, applicable for convert_model from tools.mo only
        mo_help_file = os.path.join(os.path.dirname(__file__), "mo_convert_help.py")
        params = [sys.executable, mo_help_file]
        _, mo_output, _ = shell(params)

        legacy_params = legacy_mo_params()
        for group in legacy_params:
            for param_name in group:
                assert param_name in mo_output
