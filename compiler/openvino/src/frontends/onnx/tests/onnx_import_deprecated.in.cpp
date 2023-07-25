// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "onnx_import/onnx.hpp"

using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_affine) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/affine.onnx"));

    // input/output shape (1, 3)
    auto input = test::NDArray<float, 2>{{{0.f, 1.f, 2.f}}}.get_vector();
    auto expected_output = test::NDArray<float, 2>{{{50.f, 50.5f, 51.f}}}.get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(Shape{1, 3}, input);
    test_case.add_expected_output(Shape{1, 3}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_crop) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/crop.onnx"));

    // input shape (1, 1, 4, 4)
    auto input = test::NDArray<float, 4>({{{{19.f, 20.f, 21.f, 22.f},
                                            {23.f, 24.f, 25.f, 26.f},
                                            {27.f, 28.f, 29.f, 30.f},
                                            {31.f, 32.f, 33.f, 34.f}}}})
                     .get_vector();

    // output shape (1, 1, 2, 2)
    auto expected_output = test::NDArray<float, 4>{{{{24.f, 25.f}, {28.f, 29.f}}}}.get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(Shape{1, 1, 4, 4}, input);
    test_case.add_expected_output(Shape{1, 1, 2, 2}, expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_crop_with_scale) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/crop_with_scale.onnx"));

    // input shape (1, 1, 4, 4)
    auto input = test::NDArray<float, 4>({{{{19.f, 20.f, 21.f, 22.f},
                                            {23.f, 24.f, 25.f, 26.f},
                                            {27.f, 28.f, 29.f, 30.f},
                                            {31.f, 32.f, 33.f, 34.f}}}})
                     .get_vector();

    // output shape (1, 1, 2, 3)
    auto expected_output = test::NDArray<float, 4>{{{{24.f, 25.f, 26.f}, {28.f, 29.f, 30.f}}}}.get_vector();

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input(Shape{1, 1, 4, 4}, input);
    test_case.add_expected_output(Shape{1, 1, 2, 3}, expected_output);
    test_case.run();
}
