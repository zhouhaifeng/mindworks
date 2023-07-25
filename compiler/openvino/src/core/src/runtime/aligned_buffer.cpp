// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/aligned_buffer.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

runtime::AlignedBuffer::AlignedBuffer() : m_allocated_buffer(nullptr), m_aligned_buffer(nullptr), m_byte_size(0) {}

runtime::AlignedBuffer::AlignedBuffer(size_t byte_size, size_t alignment) : m_byte_size(byte_size) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    m_byte_size = std::max<size_t>(1, byte_size);
    size_t allocation_size = m_byte_size + alignment;
    m_allocated_buffer = static_cast<char*>(ngraph_malloc(allocation_size));
    m_aligned_buffer = m_allocated_buffer;
    size_t mod = (alignment != 0) ? size_t(m_aligned_buffer) % alignment : 0;

    if (mod != 0) {
        m_aligned_buffer += (alignment - mod);
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

runtime::AlignedBuffer::AlignedBuffer(AlignedBuffer&& other)
    : m_allocated_buffer(other.m_allocated_buffer),
      m_aligned_buffer(other.m_aligned_buffer),
      m_byte_size(other.m_byte_size) {
    other.m_allocated_buffer = nullptr;
    other.m_aligned_buffer = nullptr;
    other.m_byte_size = 0;
}

runtime::AlignedBuffer::~AlignedBuffer() {
    if (m_allocated_buffer != nullptr) {
        free(m_allocated_buffer);
    }
}

runtime::AlignedBuffer& runtime::AlignedBuffer::operator=(AlignedBuffer&& other) {
    if (this != &other) {
        if (m_allocated_buffer != nullptr) {
            free(m_allocated_buffer);
        }
        m_allocated_buffer = other.m_allocated_buffer;
        m_aligned_buffer = other.m_aligned_buffer;
        m_byte_size = other.m_byte_size;
        other.m_allocated_buffer = nullptr;
        other.m_aligned_buffer = nullptr;
        other.m_byte_size = 0;
    }
    return *this;
}

namespace ov {
AttributeAdapter<shared_ptr<ngraph::runtime::AlignedBuffer>>::AttributeAdapter(
    shared_ptr<ngraph::runtime::AlignedBuffer>& value)
    : DirectValueAccessor<shared_ptr<ngraph::runtime::AlignedBuffer>>(value) {}
}  // namespace ov
