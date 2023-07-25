// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_port.hpp"

#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

ExpressionPort::ExpressionPort(const std::shared_ptr<Expression>& expr, Type type, size_t port)
        : m_expr(expr), m_type(type), m_port_index(port) {}

const PortDescriptorPtr& ExpressionPort::get_descriptor_ptr() const {
    const auto& descs = m_type == Type::Input ? m_expr->m_input_port_descriptors
                                              : m_expr->m_output_port_descriptors;
    OPENVINO_ASSERT(m_port_index < descs.size(), "Incorrect index of port");
    return descs[m_port_index];
}

const std::shared_ptr<PortConnector>& ExpressionPort::get_port_connector_ptr() const {
    const auto& connectors = m_type == Type::Input ? m_expr->m_input_port_connectors
                                                : m_expr->m_output_port_connectors;
    OPENVINO_ASSERT(m_port_index < connectors.size(), "Incorrect index of port");
    return connectors[m_port_index];
}

std::set<ExpressionPort> ExpressionPort::get_connected_ports() const {
    if (ExpressionPort::m_type == Type::Input) {
        return { m_expr->m_input_port_connectors[m_port_index]->get_source() };
    }
    if (ExpressionPort::m_type == Type::Output) {
        return m_expr->m_output_port_connectors[m_port_index]->get_consumers();
    }
    OPENVINO_THROW("ExpressionPort supports only Input and Output types");
}

bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    if (&lhs == &rhs)
        return true;
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect ExpressionPort comparison");
    return lhs.get_index() == rhs.get_index() && lhs.get_expr() == rhs.get_expr();
}
bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    return !(lhs == rhs);
}
bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect ExpressionPort comparison");
    return (lhs.get_index() < rhs.get_index()) || (lhs.get_index() == rhs.get_index() && lhs.get_expr() < rhs.get_expr());
}

}// namespace lowered
}// namespace snippets
}// namespace ov
