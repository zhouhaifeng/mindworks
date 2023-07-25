// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate_diff.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ov::operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff) {
    s << "CoordinateDiff{";
    OPENVINO_SUPPRESS_DEPRECATED_START
    s << ngraph::join(coordinate_diff);
    OPENVINO_SUPPRESS_DEPRECATED_END
    s << "}";
    return s;
}

ov::CoordinateDiff::CoordinateDiff(const std::initializer_list<std::ptrdiff_t>& diffs)
    : std::vector<std::ptrdiff_t>(diffs) {}

ov::CoordinateDiff::CoordinateDiff(const std::vector<std::ptrdiff_t>& diffs) : std::vector<std::ptrdiff_t>(diffs) {}

ov::CoordinateDiff::CoordinateDiff(const CoordinateDiff& diffs) = default;

ov::CoordinateDiff::CoordinateDiff(size_t n, std::ptrdiff_t initial_value)
    : std::vector<std::ptrdiff_t>(n, initial_value) {}

ov::CoordinateDiff::CoordinateDiff() = default;

ov::CoordinateDiff::~CoordinateDiff() = default;

ov::CoordinateDiff& ov::CoordinateDiff::operator=(const CoordinateDiff& v) {
    static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
    return *this;
}

ov::CoordinateDiff& ov::CoordinateDiff::operator=(CoordinateDiff&& v) noexcept {
    static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
    return *this;
}
