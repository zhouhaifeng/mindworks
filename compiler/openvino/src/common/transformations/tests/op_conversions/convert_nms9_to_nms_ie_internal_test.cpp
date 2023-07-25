// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/nms_ie_internal.hpp>
#include <queue>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_9.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertPreviousNMSToNMSIEInternal) {
    {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ov::op::v0::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = ov::op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = ov::op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<ov::op::v1::NonMaxSuppression>(boxes,
                                                                   scores,
                                                                   max_output_boxes_per_class,
                                                                   iou_threshold,
                                                                   score_threshold,
                                                                   op::v1::NonMaxSuppression::BoxEncodingType::CORNER,
                                                                   true);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
        manager.register_pass<ov::pass::ConvertNMS9ToNMSIEInternal>();
        manager.register_pass<ngraph::pass::ConstantFolding>();

        // as inside test infrastructure we can not predict output names for given Function
        // we have to enable soft names comparison manually
        enable_soft_names_comparison();
    }

    {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ov::op::v0::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = ov::op::v0::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = ov::op::v0::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<ov::op::internal::NonMaxSuppressionIEInternal>(boxes,
                                                                                   scores,
                                                                                   max_output_boxes_per_class,
                                                                                   iou_threshold,
                                                                                   score_threshold,
                                                                                   0,
                                                                                   true,
                                                                                   element::i32);
        auto convert = std::make_shared<ov::op::v0::Convert>(nms->output(0), element::i64);

        function_ref = std::make_shared<Function>(NodeVector{convert}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS9ToNMSIEInternal) {
    {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ov::op::v0::Constant::create(element::i32, Shape{}, {10});
        auto iou_threshold = ov::op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = ov::op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ov::op::v0::Constant::create(element::f32, Shape{}, {0.5});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_nms_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true,
                                                               element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS9ToNMSIEInternal>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto boxes = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = ov::op::v0::Constant::create(element::i32, Shape{1}, {10});
        auto iou_threshold = ov::op::v0::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = ov::op::v0::Constant::create(element::f32, Shape{1}, {0.7});
        auto soft_nms_sigma = ov::op::v0::Constant::create(element::f32, Shape{1}, {0.5});
        auto nms = std::make_shared<ov::op::internal::NonMaxSuppressionIEInternal>(boxes,
                                                                                   scores,
                                                                                   max_output_boxes_per_class,
                                                                                   iou_threshold,
                                                                                   score_threshold,
                                                                                   soft_nms_sigma,
                                                                                   0,
                                                                                   true,
                                                                                   element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}
