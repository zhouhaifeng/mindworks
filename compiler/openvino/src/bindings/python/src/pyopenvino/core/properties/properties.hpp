// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "pyopenvino/core/properties/properties.hpp"

namespace py = pybind11;

template <typename T>
void wrap_property_RO(py::module m, ov::Property<T, ov::PropertyMutability::RO> property, std::string func_name) {
    m.def(func_name.c_str(), [property]() {
        return property.name();
    });
}

template <typename T>
void wrap_property_RW(py::module m, ov::Property<T, ov::PropertyMutability::RW> property, std::string func_name) {
    m.def(func_name.c_str(), [property]() {
        return property.name();
    });

    m.def(func_name.c_str(), [property](T value) {
        return property(value);
    });
}

void regmodule_properties(py::module m);
