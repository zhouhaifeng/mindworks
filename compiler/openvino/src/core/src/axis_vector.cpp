// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/axis_vector.hpp"

#include "ngraph/util.hpp"

std::ostream& ov::operator<<(std::ostream& s, const AxisVector& axis_vector) {
    s << "AxisVector{";
    OPENVINO_SUPPRESS_DEPRECATED_START
    s << ngraph::join(axis_vector);
    OPENVINO_SUPPRESS_DEPRECATED_END
    s << "}";
    return s;
}

ov::AxisVector::AxisVector(const std::initializer_list<size_t>& axes) : std::vector<size_t>(axes) {}

ov::AxisVector::AxisVector(const std::vector<size_t>& axes) : std::vector<size_t>(axes) {}

ov::AxisVector::AxisVector(const AxisVector& axes) : std::vector<size_t>(axes) {}

ov::AxisVector::AxisVector(size_t n) : std::vector<size_t>(n) {}

ov::AxisVector::AxisVector() {}

ov::AxisVector::~AxisVector() {}

ov::AxisVector& ov::AxisVector::operator=(const AxisVector& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::AxisVector& ov::AxisVector::operator=(AxisVector&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}
