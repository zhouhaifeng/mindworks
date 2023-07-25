# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

TEST_STATUS = {
    'passed': ["[       OK ]"],
    'failed': ["[  FAILED  ]"],
    'hanged': ["Test finished by timeout"],
    'crashed': ["Unexpected application crash with code", "Segmentation fault", "Crash happens", "core dumped"],
    'skipped': ["[  SKIPPED ]"],
    'interapted': ["interapted", "Killed"]}
RUN = "[ RUN      ]"
GTEST_FILTER = "Google Test filter = "
DISABLED_PREFIX = "DISABLED_"
PG_ERR  = "PG ERROR"
PG_WARN = "PG WARN"
REF_COEF = "[ CONFORMANCE ] Influence coefficient: "

IS_WIN = "windows" in platform or "win32" in platform

OS_SCRIPT_EXT = ".bat" if IS_WIN else ""
OS_BIN_FILE_EXT = ".exe" if IS_WIN else ""
ENV_SEPARATOR = ";" if IS_WIN else ":"
PYTHON_NAME = "python" if IS_WIN else "python3"
PIP_NAME = "pip" if IS_WIN else "pip3"
LD_LIB_PATH_NAME = "PATH" if IS_WIN else "LD_LIBRARY_PATH"

OPENVINO_NAME = 'openvino'
PY_OPENVINO = "python_api"

DEBUG_DIR = "Debug"
RELEASE_DIR = "Release"

OP_CONFORMANCE = "OP"
API_CONFORMANCE = "API"

DEVICE_ARCHITECTURE_PROPERTY = "DEVICE_ARCHITECTURE"
FULL_DEVICE_PROPERTY = "FULL_DEVICE_NAME"
SUPPORTED_PROPERTIES = "SUPPORTED_PROPERTIES"


REL_WEIGHTS_REPLACE_STR = "REPLACE"
REL_WEIGHTS_FILENAME = f"rel_weights_{REL_WEIGHTS_REPLACE_STR}.lst"

NOT_EXIST_DEVICE = "NOT_EXIST_DEVICE"

MEM_USAGE = "MEM_USAGE="

CONVERT_OP_NAME = "Convert-1"

META_EXTENSION = ".meta"
XML_EXTENSION = ".xml"
BIN_EXTENSION = ".bin"
