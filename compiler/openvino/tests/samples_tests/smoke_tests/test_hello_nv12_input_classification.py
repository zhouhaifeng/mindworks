"""
 Copyright (C) 2018-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import pytest
import re
import sys
import logging as log
from common.samples_common_test_class import get_tests
from common.samples_common_test_class import SamplesCommonTestClass

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

test_data_fp32 = get_tests(cmd_params={'i': [os.path.join('224x224', 'dog6.yuv')],
                                       'm': [os.path.join('squeezenet1.1', 'FP32', 'squeezenet1.1.xml')],
                                       'size': ['224x224'],
				       'sample_type': ['C++', 'C'],
                                       'd': ['CPU']},
                           use_device=['d']
                           )

class TestHelloNV12Input(SamplesCommonTestClass):
    @classmethod
    def setup_class(cls):
        cls.sample_name = 'hello_nv12_input_classification'
        super().setup_class()

    @pytest.mark.parametrize("param", test_data_fp32)
    def test_hello_nv12_input_classification_fp32(self, param):
        _check_output(self, param=param)


def _check_output(self, param):
    """
    Classification_sample_async has functional and accuracy tests.
    For accuracy find in output class of detected on image object
    """

    # Run _test function, that returns stdout or 0.
    stdout = self._test(param, use_preffix=False, get_cmd_func=self.get_hello_nv12_cmd_line)
    if not stdout:
        return 0

    stdout = stdout.split('\n')

    is_ok = True
    for line in range(len(stdout)):
        if re.match('\\d+ +\\d+.\\d+$', stdout[line].replace('[ INFO ]', '').strip()) is not None:
            top1 = stdout[line].replace('[ INFO ]', '').strip().split(' ')[0]
            top1 = re.sub('\\D', '', top1)
            if '215' not in top1:
                is_ok = False
                log.error('Expected class 215, Detected class {}'.format(top1))
            break
    assert is_ok, 'Wrong top1 class'
    log.info('Accuracy passed')
