// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_factory.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <locale>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dict_attribute_visitor.hpp"
#include "ngraph/check.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/util/log.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

namespace {
class NodeFactory {
public:
    NodeFactory() {}
    NodeFactory(const std::string& opset_name) : m_opset(get_opset(opset_name)) {}

    std::shared_ptr<ov::Node> create(const std::string op_type_name,
                                     const ov::OutputVector& arguments,
                                     const py::dict& attributes = py::dict()) {
        std::shared_ptr<ov::Node> op_node = std::shared_ptr<ov::Node>(m_opset.create(op_type_name));

        NGRAPH_CHECK(op_node != nullptr, "Couldn't create operator: ", op_type_name);
        NGRAPH_CHECK(!ov::op::util::is_constant(op_node),
                     "Currently NodeFactory doesn't support Constant node: ",
                     op_type_name);

        util::DictAttributeDeserializer visitor(attributes, m_variables);

        op_node->set_arguments(arguments);
        op_node->visit_attributes(visitor);
        op_node->constructor_validate_and_infer_types();

        return op_node;
    }

    std::shared_ptr<ov::Node> create(const std::string op_type_name) {
        std::shared_ptr<ov::Node> op_node = std::shared_ptr<ov::Node>(m_opset.create(op_type_name));

        NGRAPH_CHECK(op_node != nullptr, "Couldn't create operator: ", op_type_name);
        NGRAPH_CHECK(!ov::op::util::is_constant(op_node),
                     "Currently NodeFactory doesn't support Constant node: ",
                     op_type_name);

        OPENVINO_WARN << "Empty op created! Please assign inputs and attributes and run validate() before op is used.";

        return op_node;
    }

private:
    const ov::OpSet& get_opset(std::string opset_ver) {
        std::locale loc;
        std::transform(opset_ver.begin(), opset_ver.end(), opset_ver.begin(), [&loc](char c) {
            return std::tolower(c, loc);
        });

        const auto& s_opsets = ov::get_available_opsets();

        auto it = s_opsets.find(opset_ver);
        OPENVINO_ASSERT(it != s_opsets.end(), "Unsupported opset version requested.");
        return it->second();
    }

    const ov::OpSet& m_opset = ov::get_opset12();
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> m_variables;
};
}  // namespace

void regclass_graph_NodeFactory(py::module m) {
    py::class_<NodeFactory> node_factory(m, "NodeFactory");
    node_factory.doc() = "NodeFactory creates nGraph nodes";

    node_factory.def(py::init());
    node_factory.def(py::init<std::string>());

    node_factory.def("create", [](NodeFactory& self, const std::string name) {
        return self.create(name);
    });
    node_factory.def(
        "create",
        [](NodeFactory& self, const std::string name, const ov::OutputVector& arguments, const py::dict& attributes) {
            return self.create(name, arguments, attributes);
        });

    node_factory.def("__repr__", [](const NodeFactory& self) {
        return Common::get_simple_repr(self);
    });
}
