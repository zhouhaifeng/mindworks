// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "ngraph/factory.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/op/ops.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace test {
class ValueHolder {
    template <typename T>
    T& invalid() {
        OPENVINO_THROW("Invalid type access");
    }

public:
    virtual ~ValueHolder() {}
    virtual operator bool&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator float&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator double&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::string&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator int8_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator int16_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator int32_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator int64_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator uint8_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator uint16_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator uint32_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator uint64_t&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<std::string>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<float>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<double>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<int8_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<int16_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<int32_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<int64_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<uint8_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<uint16_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<uint32_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<uint64_t>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator HostTensorPtr&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::shared_ptr<ov::Model>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator ov::PartialShape&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator ov::Dimension&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator std::shared_ptr<Variable>&() {
        OPENVINO_THROW("Invalid type access");
    }
    virtual operator ov::op::util::FrameworkNodeAttrs&() {
        OPENVINO_THROW("Invalid type access");
    }
    uint64_t get_index() {
        return m_index;
    }

protected:
    uint64_t m_index{0};
};

template <typename T>
class ValueHolderImp : public ValueHolder {
public:
    ValueHolderImp(const T& value, uint64_t index) : m_value(value) {
        m_index = index;
    }
    operator T&() override {
        return m_value;
    }

protected:
    T m_value;
};

class ValueMap {
    using map_type = std::unordered_map<std::string, std::shared_ptr<ValueHolder>>;

public:
    /// \brief Set to print serialization information
    void set_print(bool value) {
        m_print = value;
    }
    template <typename T>
    void insert(const std::string& name, const T& value) {
        std::pair<map_type::iterator, bool> result =
            m_values.insert(map_type::value_type(name, std::make_shared<ValueHolderImp<T>>(value, m_write_count++)));
        OPENVINO_ASSERT(result.second, name, " is already in use");
    }
    template <typename T>
    void insert_scalar(const std::string& name, const T& value) {
        std::pair<map_type::iterator, bool> result =
            m_values.insert(map_type::value_type(name, std::make_shared<ValueHolderImp<T>>(value, m_write_count++)));
        OPENVINO_ASSERT(result.second, name, " is already in use");
        if (m_print) {
            std::cerr << "SER: " << name << " = " << value << std::endl;
        }
    }
    template <typename T>
    void insert_vector(const std::string& name, const T& value) {
        std::pair<map_type::iterator, bool> result =
            m_values.insert(map_type::value_type(name, std::make_shared<ValueHolderImp<T>>(value, m_write_count++)));
        OPENVINO_ASSERT(result.second, name, " is already in use");
        if (m_print) {
            std::cerr << "SER: " << name << " = [";
            std::string comma = "";
            for (auto val : value) {
                std::cerr << comma << val;
                comma = ", ";
            }
            std::cerr << "]" << std::endl;
        }
    }

    std::size_t get_value_map_size() const {
        return m_values.size();
    }

    template <typename T>
    T& get(const std::string& name) {
        auto& value_holder = *m_values.at(name);
        OPENVINO_ASSERT(m_read_count++ == value_holder.get_index());
        return static_cast<T&>(*m_values.at(name));
    }

protected:
    map_type m_values;
    uint64_t m_write_count{0};
    uint64_t m_read_count{0};
    bool m_print{false};
};

class DeserializeAttributeVisitor : public AttributeVisitor {
public:
    DeserializeAttributeVisitor(ValueMap& value_map) : m_values(value_map) {}

    void on_adapter(const std::string& name, ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        adapter.set(m_values.get<std::shared_ptr<ov::Model>>(name));
    }

    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override {
        if (auto a = ::ov::as_type<::ov::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
                &adapter)) {
            auto& data = m_values.get<HostTensorPtr>(name);
            data->read(a->get()->get_ptr(), a->get()->size());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<
                       std::vector<std::shared_ptr<ov::op::util::SubGraphOp::OutputDescription>>>>(&adapter)) {
            a->set(m_values.get<std::vector<std::shared_ptr<ov::op::util::SubGraphOp::OutputDescription>>>(name));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<
                       std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>>>>(&adapter)) {
            a->set(m_values.get<std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>>>(name));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            a->set(m_values.get<ov::PartialShape>(name));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
            a->set(m_values.get<ov::Dimension>(name));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<Variable>>>(&adapter)) {
            a->set(m_values.get<std::shared_ptr<Variable>>(name));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
            a->set(m_values.get<ov::op::util::FrameworkNodeAttrs>(name));
        } else {
            OPENVINO_THROW("Attribute \"", name, "\" cannot be unmarshalled");
        }
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override {
        adapter.set(m_values.get<std::string>(name));
    };
    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override {
        adapter.set(m_values.get<bool>(name));
    };
    void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override {
        adapter.set(m_values.get<int64_t>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override {
        adapter.set(m_values.get<double>(name));
    }

    void on_adapter(const std::string& name, ValueAccessor<std::vector<int8_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<int8_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int16_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<int16_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int32_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<int32_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<int64_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint8_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<uint8_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint16_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<uint16_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint32_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<uint32_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter) override {
        adapter.set(m_values.get<std::vector<uint64_t>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<std::string>>& adapter) override {
        adapter.set(m_values.get<std::vector<std::string>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter) override {
        adapter.set(m_values.get<std::vector<float>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<double>>& adapter) override {
        adapter.set(m_values.get<std::vector<double>>(name));
    }
    void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override {
        HostTensorPtr& data = m_values.get<HostTensorPtr>(name);
        data->read(adapter.get_ptr(), adapter.size());
    }

protected:
    ValueMap& m_values;
};

class SerializeAttributeVisitor : public AttributeVisitor {
public:
    SerializeAttributeVisitor(ValueMap& value_map) : m_values(value_map) {}

    void on_adapter(const std::string& name, ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        m_values.insert(name, adapter.get());
    }

    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override {
        if (auto a = ::ov::as_type<::ov::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
                &adapter)) {
            HostTensorPtr data = std::make_shared<HostTensor>(element::u8, Shape{a->get()->size()});
            data->write(a->get()->get_ptr(), a->get()->size());
            m_values.insert(name, data);
        } else if (auto a = ov::as_type<ov::AttributeAdapter<
                       std::vector<std::shared_ptr<ov::op::util::SubGraphOp::OutputDescription>>>>(&adapter)) {
            m_values.insert_vector(name, a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<
                       std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>>>>(&adapter)) {
            m_values.insert_vector(name, a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            m_values.insert_vector(name, a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
            m_values.insert(name, a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<Variable>>>(&adapter)) {
            m_values.insert(name, a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
            m_values.insert(name, a->get());
        } else {
            OPENVINO_THROW("Attribute \"", name, "\" cannot be marshalled");
        }
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override {
        m_values.insert_scalar(name, adapter.get());
    };
    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override {
        m_values.insert_scalar(name, adapter.get());
    };

    void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override {
        m_values.insert_scalar(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override {
        m_values.insert_scalar(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<std::string>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<double>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int8_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int16_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int32_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint8_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint16_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint32_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_values.insert_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override {
        HostTensorPtr data = std::make_shared<HostTensor>(element::u8, Shape{adapter.size()});
        data->write(adapter.get_ptr(), adapter.size());
        m_values.insert(name, data);
    }

protected:
    ValueMap& m_values;
};

class NodeBuilder : public ValueMap, public DeserializeAttributeVisitor {
public:
    NodeBuilder() : DeserializeAttributeVisitor(static_cast<ValueMap&>(*this)), m_serializer(*this), m_inputs{} {}

    NodeBuilder(const std::shared_ptr<Node>& node, ov::OutputVector inputs = {})
        : DeserializeAttributeVisitor(static_cast<ValueMap&>(*this)),
          m_serializer(*this),
          m_inputs(inputs) {
        save_node(node);
    }

    void save_node(std::shared_ptr<Node> node) {
        m_node_type_info = node->get_type_info();
        node->visit_attributes(m_serializer);
    }

    std::shared_ptr<Node> create() {
        std::shared_ptr<Node> node(get_ops().create(m_node_type_info));
        node->visit_attributes(*this);

        if (m_inputs.size()) {
            node->set_arguments(m_inputs);
            return node->clone_with_new_inputs(m_inputs);
        } else {
            return node;
        }
    }

    AttributeVisitor& get_node_saver() {
        return m_serializer;
    }
    AttributeVisitor& get_node_loader() {
        return *this;
    }
    static FactoryRegistry<Node>& get_ops() {
        static FactoryRegistry<Node> registry = [] {
            FactoryRegistry<Node> registry;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) registry.register_factory<NAMESPACE::NAME>();
#include "op_version_tbl.hpp"
#undef _OPENVINO_OP_REG
            return registry;
        }();
        return registry;
    }

protected:
    Node::type_info_t m_node_type_info;
    SerializeAttributeVisitor m_serializer;
    ov::OutputVector m_inputs;
};
}  // namespace test
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
