// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/utils/ngraph_helpers.hpp"


namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType,
                                             std::vector<size_t> inShape,
                                             std::vector<float> constantsValue) {
    switch (activationType) {
        case ngraph::helpers::ActivationTypes::Sigmoid:
            return std::make_shared<ngraph::op::Sigmoid>(in);
        case ngraph::helpers::ActivationTypes::Tanh:
            return std::make_shared<ngraph::op::Tanh>(in);
        case ngraph::helpers::ActivationTypes::Relu:
            return std::make_shared<ngraph::op::Relu>(in);
        case ngraph::helpers::ActivationTypes::LeakyRelu: {
            auto leaky_slope = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::f32,
                    inShape,
                    constantsValue);
            return std::make_shared<ngraph::op::PRelu>(in, leaky_slope);
        }
        case ngraph::helpers::ActivationTypes::Exp:
            return std::make_shared<ngraph::op::Exp>(in);
        case ngraph::helpers::ActivationTypes::Log:
            return std::make_shared<ngraph::op::Log>(in);
        case ngraph::helpers::ActivationTypes::Sign:
            return std::make_shared<ngraph::op::Sign>(in);
        case ngraph::helpers::ActivationTypes::Abs:
            return std::make_shared<ngraph::op::Abs>(in);
        case ngraph::helpers::ActivationTypes::Gelu:
            return std::make_shared<ngraph::op::v0::Gelu>(in);
        case ngraph::helpers::ActivationTypes::Clamp:
            return std::make_shared<ngraph::op::Clamp>(in, constantsValue[0], constantsValue[1]);
        case ngraph::helpers::ActivationTypes::Negative:
            return std::make_shared<ngraph::op::Negative>(in);
        case ngraph::helpers::ActivationTypes::Acos:
            return std::make_shared<ngraph::op::Acos>(in);
        case ngraph::helpers::ActivationTypes::Acosh:
            return std::make_shared<ngraph::op::Acosh>(in);
        case ngraph::helpers::ActivationTypes::Asin:
            return std::make_shared<ngraph::op::Asin>(in);
        case ngraph::helpers::ActivationTypes::Asinh:
            return std::make_shared<ngraph::op::Asinh>(in);
        case ngraph::helpers::ActivationTypes::Atan:
            return std::make_shared<ngraph::op::Atan>(in);
        case ngraph::helpers::ActivationTypes::Atanh:
            return std::make_shared<ngraph::op::Atanh>(in);
        case ngraph::helpers::ActivationTypes::Cos:
            return std::make_shared<ngraph::op::Cos>(in);
        case ngraph::helpers::ActivationTypes::Cosh:
            return std::make_shared<ngraph::op::Cosh>(in);
        case ngraph::helpers::ActivationTypes::Floor:
            return std::make_shared<ngraph::op::Floor>(in);
        case ngraph::helpers::ActivationTypes::Sin:
            return std::make_shared<ngraph::op::Sin>(in);
        case ngraph::helpers::ActivationTypes::Sinh:
            return std::make_shared<ngraph::op::Sinh>(in);
        case ngraph::helpers::ActivationTypes::Sqrt:
            return std::make_shared<ngraph::op::Sqrt>(in);
        case ngraph::helpers::ActivationTypes::Tan:
            return std::make_shared<ngraph::op::Tan>(in);
        case ngraph::helpers::ActivationTypes::Elu:
            return std::make_shared<ngraph::op::Elu>(in, constantsValue[0]);
        case ngraph::helpers::ActivationTypes::Erf:
            return std::make_shared<ngraph::op::Erf>(in);
        case ngraph::helpers::ActivationTypes::HardSigmoid: {
            auto hard_sigmoid_alpha = std::make_shared<ngraph::op::Constant>(
                    type, inShape, constantsValue[0]);
            auto hard_sigmoid_beta = std::make_shared<ngraph::op::Constant>(
                    type, inShape, constantsValue[1]);
            return std::make_shared<ngraph::op::HardSigmoid>(in, hard_sigmoid_alpha, hard_sigmoid_beta);
        }
        case ngraph::helpers::ActivationTypes::Selu: {
            auto selu_alpha = std::make_shared<ngraph::op::Constant>(
                    type, inShape, constantsValue[0]);
            auto selu_lambda = std::make_shared<ngraph::op::Constant>(
                    type, inShape, constantsValue[1]);
            return std::make_shared<ngraph::op::Selu>(in, selu_alpha, selu_lambda);
        }
        case ngraph::helpers::ActivationTypes::Ceiling:
            return std::make_shared<ngraph::op::Ceiling>(in);
        case ngraph::helpers::ActivationTypes::PReLu: {
            auto negative_slope = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::f32,
                    inShape,
                    constantsValue);
            return std::make_shared<ngraph::op::PRelu>(in, negative_slope);
        }
        case ngraph::helpers::ActivationTypes::Mish:
            return std::make_shared<ngraph::op::v4::Mish>(in);
        case ngraph::helpers::ActivationTypes::HSwish:
            return std::make_shared<ngraph::op::v4::HSwish>(in);
        case ngraph::helpers::ActivationTypes::SoftPlus:
            return std::make_shared<ngraph::op::v4::SoftPlus>(in);
        case ngraph::helpers::ActivationTypes::Swish: {
            auto beta = std::make_shared<ngraph::op::Constant>(type, inShape, constantsValue[0]);
            return std::make_shared<ngraph::op::v4::Swish>(in, beta);
        }
        case ngraph::helpers::ActivationTypes::HSigmoid:
            return std::make_shared<ngraph::op::v5::HSigmoid>(in);
        case ngraph::helpers::ActivationTypes::RoundHalfToEven:
            return std::make_shared<ngraph::op::v5::Round>(in, ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN);
        case ngraph::helpers::ActivationTypes::RoundHalfAwayFromZero:
            return std::make_shared<ngraph::op::v5::Round>(in, ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
        case ngraph::helpers::ActivationTypes::GeluErf:
            return std::make_shared<ngraph::op::v7::Gelu>(in, ngraph::op::GeluApproximationMode::ERF);
        case ngraph::helpers::ActivationTypes::GeluTanh:
            return std::make_shared<ngraph::op::v7::Gelu>(in, ngraph::op::GeluApproximationMode::TANH);
        case ngraph::helpers::ActivationTypes::SoftSign:
            return std::make_shared<ngraph::op::v9::SoftSign>(in);
        default:
            throw std::runtime_error("Can't create layer for this activation type");
    }
}

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::ParameterVector &parameters,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType) {
    switch (activationType) {
        case ngraph::helpers::ActivationTypes::LeakyRelu:
            return std::make_shared<ngraph::op::PRelu>(parameters[0], parameters[1]);
        case ngraph::helpers::ActivationTypes::HardSigmoid:
            return std::make_shared<ngraph::op::HardSigmoid>(parameters[0], parameters[1], parameters[2]);
        case ngraph::helpers::ActivationTypes::Selu:
            return std::make_shared<ngraph::op::Selu>(parameters[0], parameters[1], parameters[2]);
        case ngraph::helpers::ActivationTypes::PReLu:
            return std::make_shared<ngraph::op::PRelu>(parameters[0], parameters[1]);
        default:
            throw std::runtime_error("It is impossible to create layer for this activation type with input as parameter");
    }
}

}  // namespace builder
}  // namespace ngraph
