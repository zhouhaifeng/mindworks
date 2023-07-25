// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "ie_extension.h"
#include <condition_variable>
#include "openvino/core/shape.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "transformations/utils/utils.hpp"
#include <string>
#include <ie_core.hpp>
#include <thread>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "shared_test_classes/subgraph/basic_lstm.hpp"
#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestDynamicTests::getTestCaseName(testing::TestParamInfo<OVInferRequestDynamicParams> obj) {
    std::shared_ptr<Model> func;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
    std::string target_device;
    ov::AnyMap configuration;
    std::tie(func, inOutShapes, target_device, configuration) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "function=" << func->get_friendly_name() << "_";
    result << "inOutShape=(";
    for (const auto& inOutShape : inOutShapes) {
        result << "(" << CommonTestUtils::vec2str(inOutShape.first) << "_" << CommonTestUtils::vec2str(inOutShape.second) << ")";
    }
    result << ")_";
    result << "targetDevice=" << target_device << "_";
    if (!configuration.empty()) {
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
    }
    return result.str();
}

void OVInferRequestDynamicTests::SetUp() {
    std::tie(function, inOutShapes, target_device, configuration) = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
}

bool OVInferRequestDynamicTests::checkOutput(const ov::runtime::Tensor& in, const ov::runtime::Tensor& actual) {
    bool result = true;
    auto net = ie->compile_model(function, CommonTestUtils::DEVICE_TEMPLATE);
    ov::InferRequest req;
    req = net.create_infer_request();
    auto tensor = req.get_tensor(function->inputs().back().get_any_name());
    tensor.set_shape(in.get_shape());
    for (int i = 0; i < in.get_size(); i++) {
        tensor.data<float>()[i] = in.data<float>()[i];
    }
    req.infer();
    for (int i = 0; i < actual.get_size(); i++) {
        if (fabs(req.get_output_tensor(0).data<float>()[i] - actual.data<float>()[i]) > std::numeric_limits<float>::epsilon())
            return false;
    }
    return result;
}

/*
We have to check that we don't get a segmentation fault during
inference if we set the first two times to the same shape and
then a different one for the case with upper bounds.

Previously, this resulted in a segmentation fault for the CPU plugin.
*/
TEST_P(OVInferRequestDynamicTests, InferDynamicNetwork) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].first, inOutShapes[1].first};
    const std::string tensor_name = "input_tensor";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = { ov::Dimension(1, inOutShapes[1].first[0]),
                            ov::Dimension(1, inOutShapes[1].first[1]),
                            ov::Dimension(1, inOutShapes[1].first[2]),
                            ov::Dimension(1, inOutShapes[1].first[3])
    };
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    const std::string outputname = function->outputs().back().get_any_name();
    for (auto& shape : vectorShapes) {
        ov::runtime::Tensor inTensor = ov::test::utils::create_and_fill_tensor(element::f32, shape, 100, -50);
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        OV_ASSERT_NO_THROW(req.set_tensor("input_tensor", inTensor));
        OV_ASSERT_NO_THROW(req.infer());
        ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputname)));
    }
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkSetUnexpectedOutputTensorBeforeInfer) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::runtime::Tensor tensor, otensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    tensor = ov::test::utils::create_and_fill_tensor(element::f32, refShape, 100, -50);
    OV_ASSERT_NO_THROW(req.set_tensor("input_tensor", tensor));
    auto outShape = refOutShape;
    outShape[0] += 1;
    otensor = ov::test::utils::create_and_fill_tensor(element::f32, outShape, 100, 50);
    OV_ASSERT_NO_THROW(req.set_tensor(outputname, otensor));
    OV_ASSERT_NO_THROW(req.infer());
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputname)));
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkSetOutputTensorPreAllocatedMemoryBeforeInfer) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::runtime::Tensor tensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    tensor = ov::test::utils::create_and_fill_tensor(element::f32, refShape, 100, -50);
    OV_ASSERT_NO_THROW(req.set_tensor("input_tensor", tensor));
    float ptr[5000];
    ov::runtime::Tensor otensor(element::f32, refOutShape, ptr);
    OV_ASSERT_NO_THROW(req.set_tensor(outputname, otensor));
    OV_ASSERT_NO_THROW(req.infer());
    ASSERT_EQ(req.get_tensor(outputname).data<float>(), ptr);
    ASSERT_EQ(req.get_tensor(outputname).get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputname)));
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkSetOutputShapeBeforeInfer) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::runtime::Tensor tensor, otensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    tensor = ov::test::utils::create_and_fill_tensor(element::f32, refShape, 100, -50);
    OV_ASSERT_NO_THROW(req.set_tensor("input_tensor", tensor));
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    OV_ASSERT_NO_THROW(otensor.set_shape(refOutShape));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputname)));
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithoutSetShape) {
    const std::string tensor_name = "input_tensor";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkBoundWithoutSetShape) {
    const std::string tensor_name = "input_tensor";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(0, 5), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
}


TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithGetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor, otensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    //OV_ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    EXPECT_NE(0, otensor.get_size()); // output tensor is allocated after infer
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputname)));
}

TEST_P(OVInferRequestDynamicTests, InferUpperBoundNetworkWithGetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(0, 19), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor, otensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    //OV_ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputname)));
}

TEST_P(OVInferRequestDynamicTests, InferUpperBoundNetworkAfterIOTensorsReshaping) {
    const std::string tensor_name = "input_tensor";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(0, 19), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor, otensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    OV_ASSERT_NO_THROW(otensor.set_shape({1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(otensor.set_shape({4, 4, 20, 20}));
    OV_ASSERT_NO_THROW(otensor.set_shape({1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(tensor.set_shape({4, 4, 20, 20}));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestDynamicTests, InferFullyDynamicNetworkWithGetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = ov::PartialShape::dynamic();
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor, otensor;
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    //OV_ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputName));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputName));
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputName)));
}

TEST_P(OVInferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetTensorLower) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(2, 3), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    //ASSERT_THROW(req.infer(), ov::Exception);
}

TEST_P(OVInferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetTensorUpper) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(1, 2), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({3, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    // ASSERT_THROW(req.infer(), ov::Exception);
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithGetTensor2times) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape(refShape));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    req.wait();
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputName)));

    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape(refShape2));
    ASSERT_EQ(tensor.get_shape(), refShape2);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    req.wait();
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape2);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputName)));
}


TEST_P(OVInferRequestDynamicTests, GetSameTensor2times) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape(refShape));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    ASSERT_EQ(tensor.get_shape(), refShape);
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithSetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor(ov::element::f32, refShape);
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputName)));
}

TEST_P(OVInferRequestDynamicTests, InferFullyDynamicNetworkWithSetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = ov::PartialShape::dynamic();
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor(ov::element::f32, refShape), otensor;
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputName));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(req.infer());
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputName)));
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithSetTensor2times) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    const std::string outputName = function->outputs().back().get_any_name();
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor(ov::element::f32, refShape);

    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputName)));

    tensor = ov::Tensor(ov::element::f32, refShape2);
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape2);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape2);
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor"), req.get_tensor(outputName)));
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithLocalCore) {
    ov::CompiledModel compiled_model;
    {
        ov::Core local_core = createCoreWithTemplate();
        const std::string tensor_name = "input_tensor";
        std::map<std::string, ov::PartialShape> shapes;
        shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
        OV_ASSERT_NO_THROW(function->reshape(shapes));
        // Load ov::Model to target plugins
        compiled_model = local_core.compile_model(function, target_device, configuration);
    }
    // Create InferRequest
    OV_ASSERT_NO_THROW(compiled_model.create_infer_request());
}

TEST_P(OVNotSupportRequestDynamicTests, InferDynamicNotSupported) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    const std::string outputName = function->outputs().back().get_any_name();
    // Load ov::Function to target plugins
    ov::CompiledModel execNet;
    ASSERT_THROW((execNet = ie->compile_model(function, target_device, configuration)), ov::Exception);
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
