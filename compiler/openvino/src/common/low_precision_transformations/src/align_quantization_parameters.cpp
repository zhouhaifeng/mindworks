// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/align_quantization_parameters.hpp"
#include <memory>
#include "low_precision/create_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_granularity_attribute.hpp"
#include "low_precision/update_shared_precision_preserved.hpp"
#include "itt.hpp"
#include "openvino/pass/manager.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

ngraph::pass::low_precision::AlignQuantizationParameters::AlignQuantizationParameters(const std::vector<ngraph::element::Type> defaultPrecisions)
    : defaultPrecisions(defaultPrecisions) {}

bool ngraph::pass::low_precision::AlignQuantizationParameters::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(AlignQuantizationParameters);
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    std::shared_ptr<ngraph::pass::GraphRewrite> propagation = manager.register_pass<ngraph::pass::GraphRewrite>();
    propagation->add_matcher<low_precision::CreateAttribute<QuantizationAlignmentAttribute>>();
    propagation->add_matcher<low_precision::PropagateThroughPrecisionPreserved<QuantizationAlignmentAttribute>>();
    propagation->add_matcher<low_precision::UpdateSharedPrecisionPreserved<QuantizationAlignmentAttribute, QuantizationGranularityAttribute>>();
    manager.run_passes(f);
    return false;
}
