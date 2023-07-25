// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ov::test;

namespace SubgraphTestsDefinitions {

class FuseNon0OuputPort : public SubgraphBaseTest {
    void SetUp() override {
        const ov::Shape x_shape = {1, 10};
        const ov::Shape y_shape = {1};
        const ov::Shape z_shape = {1};
        ngraph::ParameterVector params(3);
        targetStaticShapes = {{x_shape, y_shape, z_shape}};
        targetDevice = CommonTestUtils::DEVICE_CPU;
        params[0] = ngraph::builder::makeParams(ov::element::f32, {x_shape})[0];
        params[1] = ngraph::builder::makeParams(ov::element::i32, {y_shape})[0];
        params[2] = ngraph::builder::makeParams(ov::element::i32, {z_shape})[0];

        // make a sub function
        const auto cond = ov::op::v0::Constant::create(ov::element::boolean, {1}, {true});
        ngraph::ParameterVector sub_params(3);
        sub_params[0] = ngraph::builder::makeParams(ov::element::f32, {x_shape})[0];
        sub_params[1] = ngraph::builder::makeParams(ov::element::i32, {y_shape})[0];
        sub_params[2] = ngraph::builder::makeParams(ov::element::boolean, {y_shape})[0];
        ngraph::ResultVector sub_results(3);
        sub_results[0] = std::make_shared<ngraph::opset1::Result>(sub_params[0]);
        sub_results[1] = std::make_shared<ngraph::opset1::Result>(sub_params[1]);
        sub_results[2] = std::make_shared<ngraph::opset1::Result>(sub_params[2]);
        const auto sub_model = std::make_shared<ov::Model>(sub_results, sub_params);

        // loop ops
        const auto trip = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        const auto loop = std::make_shared<ov::op::v5::Loop>(trip, cond);
        loop->set_function(sub_model);
        loop->set_invariant_input(sub_params[0], params[0]);
        loop->set_invariant_input(sub_params[1], params[1]);
        loop->set_invariant_input(sub_params[2], cond);
        loop->set_special_body_ports({-1, 2});
        const auto out0 = loop->get_iter_value(sub_results[0]->output(0), -1);
        const auto out1 = loop->get_iter_value(sub_results[1]->output(0), -1);
        const auto out2 = loop->get_iter_value(sub_results[2]->output(0), -1);

        // main function
        const auto c = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
        const auto z1 = std::make_shared<ov::op::v1::Add>(params[2], c);
        const auto d = std::make_shared<ov::op::v1::Add>(out1, z1);
        function = std::make_shared<ov::Model>(ov::OutputVector{d->output(0), out0, out2}, params, "FuseNon0OuputPort");
    }
};

TEST_F(FuseNon0OuputPort, smoke_FuseNon0OuputPort) {
    run();
}

} // namespace SubgraphTestsDefinitions
