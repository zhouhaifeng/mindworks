// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/axis_vector.hpp"  // ngraph::AxisVector

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyngraph/axis_vector.hpp"

namespace py = pybind11;

void regclass_pyngraph_AxisVector(py::module m) {
    py::class_<ngraph::AxisVector, std::shared_ptr<ngraph::AxisVector>> axis_vector(m,
                                                                                    "AxisVector",
                                                                                    py::module_local());
    axis_vector.doc() = "ngraph.impl.AxisVector wraps ngraph::AxisVector";
    axis_vector.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axes"));
    axis_vector.def(py::init<const std::vector<size_t>&>(), py::arg("axes"));
    axis_vector.def(py::init<const ngraph::AxisVector&>(), py::arg("axes"));
}
