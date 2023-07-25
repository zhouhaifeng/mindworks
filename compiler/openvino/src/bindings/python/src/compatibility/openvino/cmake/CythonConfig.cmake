# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
# Following changes were done on top of original file:
# Add CYTHON_EXECUTABLE searching hints at lines 50 and 51

#=============================================================================
# Copyright 2011 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
# Find the Cython compiler.
#
# This code sets the following variables:
#
#  CYTHON_EXECUTABLE
#
# See also UseCython.cmake
# Use the Cython executable that lives next to the Python executable
# if it is a local installation.

find_package(PythonInterp 3 QUIET)
if( PYTHONINTERP_FOUND )
  get_filename_component( _python_path ${PYTHON_EXECUTABLE} PATH )
  file(TO_CMAKE_PATH "$ENV{HOME}" ENV_HOME)
  find_host_program( CYTHON_EXECUTABLE
    NAMES cython cython.bat cython3
    HINTS ${_python_path} ${ENV_HOME}/.local/bin $ENV{HOMEBREW_OPT}/cython/bin 
          ${ENV_HOME}/Library/Python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/bin
    )
else()
  find_host_program( CYTHON_EXECUTABLE
    NAMES cython cython.bat cython3
    )
endif()


include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( Cython REQUIRED_VARS CYTHON_EXECUTABLE )

# Find Cython version
execute_process(COMMAND ${CYTHON_EXECUTABLE} -V ERROR_VARIABLE CYTHON_OUTPUT OUTPUT_QUIET)
string(REGEX REPLACE "^Cython version ([0-9]+\\.[0-9]+(\\.[0-9]+)?).*" "\\1" CYTHON_VERSION "${CYTHON_OUTPUT}")

mark_as_advanced( CYTHON_EXECUTABLE CYTHON_VERSION )
