// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>

#include "paddle_utils.hpp"
#include "utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

TEST(FrontEndIncorrectCutModelTest, test_incorrect_cut) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(PADDLE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) +
                                                             std::string("2in_2out/2in_2out.pdmodel"));
    ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);

    // remove second input
    inputModel->override_all_inputs({inputModel->get_inputs()[0]});

    std::shared_ptr<ngraph::Function> function;
    ASSERT_THROW(function = frontEnd->convert(inputModel), GeneralFailure);
    ASSERT_EQ(function, nullptr);
}
