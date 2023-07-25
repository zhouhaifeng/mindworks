// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/precision_preserved_attribute.hpp"

#include <memory>
#include <string>

using namespace ngraph;
using namespace ov;

PrecisionPreservedAttribute::PrecisionPreservedAttribute(const bool value) :
    SharedAttribute(value) {
}

std::string PrecisionPreservedAttribute::to_string() const {
    std::stringstream ss;
    ss << attribute->get_string();
    ss << "value: " << (value() ? "true" : "false");
    return ss.str();
}
