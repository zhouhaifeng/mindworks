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
#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/opsets/opset.hpp"
#include "openvino/opsets/opset.hpp"

namespace py = pybind11;

namespace {
class NodeFactory {
public:
    NodeFactory() {}
    NodeFactory(const std::string& opset_name) : m_opset(get_opset(opset_name)) {}

    std::shared_ptr<ngraph::Node> create(const std::string op_type_name,
                                         const ngraph::OutputVector& arguments,
                                         const py::dict& attributes = py::dict()) {
        std::shared_ptr<ngraph::Node> op_node = std::shared_ptr<ngraph::Node>(m_opset.create(op_type_name));

        NGRAPH_CHECK(op_node != nullptr, "Couldn't create operator: ", op_type_name);
        NGRAPH_CHECK(!ngraph::op::is_constant(op_node),
                     "Currently NodeFactory doesn't support Constant node: ",
                     op_type_name);

        util::DictAttributeDeserializer visitor(attributes, m_variables);

        op_node->set_arguments(arguments);
        op_node->visit_attributes(visitor);
        op_node->constructor_validate_and_infer_types();

        return op_node;
    }

    std::shared_ptr<ngraph::Node> create(const std::string op_type_name) {
        std::shared_ptr<ngraph::Node> op_node = std::shared_ptr<ngraph::Node>(m_opset.create(op_type_name));

        NGRAPH_CHECK(op_node != nullptr, "Couldn't create operator: ", op_type_name);
        NGRAPH_CHECK(!ngraph::op::is_constant(op_node),
                     "Currently NodeFactory doesn't support Constant node: ",
                     op_type_name);

        NGRAPH_WARN << "Empty op created! Please assign inputs and attributes and run validate() before op is used.";

        return op_node;
    }

private:
    const ngraph::OpSet& get_opset(std::string opset_ver) {
        std::locale loc;
        std::transform(opset_ver.begin(), opset_ver.end(), opset_ver.begin(), [&loc](char c) {
            return std::tolower(c, loc);
        });

        const auto& s_opsets = ngraph::get_available_opsets();

        auto it = s_opsets.find(opset_ver);
        if (it == s_opsets.end()) {
            OPENVINO_THROW("Unsupported opset version requested.");
        }
        return it->second();
    }

    const ngraph::OpSet& m_opset = ngraph::get_opset11();
    std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>> m_variables;
};
}  // namespace

void regclass_pyngraph_NodeFactory(py::module m) {
    py::class_<NodeFactory> node_factory(m, "NodeFactory", py::module_local());
    node_factory.doc() = "NodeFactory creates nGraph nodes";

    node_factory.def(py::init());
    node_factory.def(py::init<std::string>());

    node_factory.def("create", [](NodeFactory& self, const std::string name) {
        return self.create(name);
    });
    node_factory.def("create",
                     [](NodeFactory& self,
                        const std::string name,
                        const ngraph::OutputVector& arguments,
                        const py::dict& attributes) {
                         return self.create(name, arguments, attributes);
                     });

    node_factory.def("__repr__", [](const NodeFactory& self) {
        return "<NodeFactory>";
    });
}
