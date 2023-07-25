// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace functional {

// todo: reuse in summary
inline std::string get_node_version(const std::shared_ptr<ov::Node>& node, const std::string& postfix = "") {
    std::string op_name = node->get_type_info().name;
    std::string opset_version = node->get_type_info().get_version();
    std::string opset_name = "opset";
    auto pos = opset_version.find(opset_name);
    if (pos != std::string::npos) {
        op_name +=  "-" + opset_version.substr(pos + opset_name.size());
    }
    if (!postfix.empty()) {
        op_name += "_" + postfix;
    }
    return op_name;
}

}  // namespace functional
}  // namespace test
}  // namespace ov

namespace LayerTestsUtils {

struct ModelInfo {
    size_t unique_op_cnt;
    // model_path, op_cnt
    std::map<std::string, size_t> model_paths;


    ModelInfo(size_t _op_cnt = 0, const std::map<std::string, size_t>& _model_paths = {{}})
        : unique_op_cnt(_op_cnt),
          model_paths(_model_paths) {}
};

struct PortInfo {
    double min;
    double max;
    bool convert_to_const;

    PortInfo(double min, double max, bool convert_to_const) : min(min), max(max),
                                                              convert_to_const(convert_to_const) {}
    PortInfo() {
        min = std::numeric_limits<double>::min();
        max = std::numeric_limits<double>::max();
        convert_to_const = false;
    }
};

struct OPInfo {
    std::map<std::string, ModelInfo> found_in_models;
    std::map<size_t, PortInfo> ports_info;

    OPInfo(const std::string& source_model, const std::string& model_path, size_t total_op_cnt = 0) {
        found_in_models = {{source_model, ModelInfo(1, {{model_path, total_op_cnt}})}};
        ports_info = {};
    }

    OPInfo() = default;
};
} // namespace LayerTestsUtils
