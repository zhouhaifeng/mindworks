// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/attribute_visitor.hpp"


namespace ov {
namespace snippets {
namespace lowered {

class PortDescriptor;
using PortDescriptorPtr = std::shared_ptr<PortDescriptor>;
class PortDescriptor {
public:
    // The structure with service values for scheduling parameters
    struct ServiceDimensions {
        // The value for the subtensor that means that scheduling should be by full dimension
        static size_t FULL_DIM;
    };

    explicit PortDescriptor(const ov::Input<ov::Node>& node,
                            std::vector<size_t> subtensor_shape = {},
                            std::vector<size_t> layout = {});
    explicit PortDescriptor(const ov::Input<const ov::Node>& node,
                            std::vector<size_t> subtensor_shape = {},
                            std::vector<size_t> layout = {});
    explicit PortDescriptor(const ov::Output<ov::Node>& node,
                            std::vector<size_t> subtensor_shape = {},
                            std::vector<size_t> layout = {});
    explicit PortDescriptor(const ov::Output<const ov::Node>& node,
                            std::vector<size_t> subtensor_shape = {},
                            std::vector<size_t> layout = {});
    PortDescriptor(std::vector<size_t> shape, std::vector<size_t> subtensor_shape, std::vector<size_t> layout = {});
    PortDescriptor() = default;

    std::vector<size_t> get_shape() const {return m_tensor_shape;}
    std::vector<size_t> get_subtensor() const {return m_subtensor_shape;}
    std::vector<size_t> get_layout() const {return m_layout;}
    size_t get_reg() const { return m_reg; }

    void set_shape(const std::vector<size_t>& tensor) { m_tensor_shape = tensor; }
    void set_layout(const std::vector<size_t>& layout) { m_layout = layout; }
    void set_subtensor(const std::vector<size_t>& subtensor) { m_subtensor_shape = subtensor; }
    void set_reg(size_t reg) { m_reg = reg; }

    std::string serialize() const;
    bool empty() const { return m_layout.empty() && m_subtensor_shape.empty();}
    PortDescriptorPtr clone() const;

    friend bool operator==(const PortDescriptor& lhs, const PortDescriptor& rhs);
    friend bool operator!=(const PortDescriptor& lhs, const PortDescriptor& rhs) {return !(lhs == rhs);}

private:
    void validate_arguments();
    /// \brief Original tensor shape
    std::vector<size_t> m_tensor_shape{};
    /// \brief Order of dimensions: NCHW == {0, 1, 2, 3}, NHWC == {0, 2, 3, 1}, NCHW16c == {0, 1, 2, 3, 1}
    std::vector<size_t> m_layout{};
    /// \brief Minimal tensor size that could be processed in one call
    std::vector<size_t> m_subtensor_shape{};
    /// \brief The corresponding abstract/physical register
    size_t m_reg = 0;
};

class PortDescriptorUtils {
public:
    static void set_port_descriptor_ptr(const ov::Input<ov::Node>& n, const PortDescriptorPtr& desc);
    static void set_port_descriptor_ptr(const ov::Output<ov::Node>& n, const PortDescriptorPtr& desc);

    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Input<ov::Node>& in);
    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Input<const ov::Node>& out);
    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Output<ov::Node>& in);
    static PortDescriptorPtr get_port_descriptor_ptr(const ov::Output<const ov::Node>& out);

    static void clean(const std::shared_ptr<ov::Node>& node);

private:
    static void init_default(std::vector<PortDescriptorPtr>& in_descs, std::vector<PortDescriptorPtr>& out_descs, const std::shared_ptr<ov::Node>& node);
};

// PortDescriptorVectorAttribute is not copyable attribute!
// It's needed to avoid incorrect copies of rt info between different nodes in call copy_runtime_info() (for example, in transformations)
// The attribute must be manually copied if needed
class PortDescriptorVectorAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("PortDescriptorVectorAttribute", "", ov::RuntimeAttribute);

    PortDescriptorVectorAttribute() = default;
    explicit PortDescriptorVectorAttribute(std::vector<PortDescriptorPtr> in_descs = {}, std::vector<PortDescriptorPtr> out_descs = {})
            : inputs(std::move(in_descs)), outputs(std::move(out_descs)) {}

    bool is_copyable() const override { return false; }

    std::vector<PortDescriptorPtr> inputs{};
    std::vector<PortDescriptorPtr> outputs{};
};

} // namespace lowered
} // namespace snippets
} // namespace ov
