// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/strides.hpp"  // ngraph::Strides

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyngraph/strides.hpp"

namespace py = pybind11;

void regclass_pyngraph_Strides(py::module m) {
    py::class_<ngraph::Strides, std::shared_ptr<ngraph::Strides>> strides(m, "Strides", py::module_local());
    strides.doc() = "ngraph.impl.Strides wraps ngraph::Strides";
    strides.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axis_strides"));
    strides.def(py::init<const std::vector<size_t>&>(), py::arg("axis_strides"));
    strides.def(py::init<const ngraph::Strides&>(), py::arg("axis_strides"));

    strides.def("__str__", [](const ngraph::Strides& self) -> std::string {
        std::stringstream stringstream;
        std::copy(self.begin(), self.end(), std::ostream_iterator<size_t>(stringstream, ", "));
        std::string string = stringstream.str();
        return string.substr(0, string.size() - 2);
    });

    strides.def("__repr__", [](const ngraph::Strides& self) -> std::string {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::string shape_str = py::cast(self).attr("__str__")().cast<std::string>();
        return "<" + class_name + ": (" + shape_str + ")>";
    });
}
