// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/tensor_external_data.hpp"

#include <fstream>
#include <sstream>

#include "exceptions.hpp"
#include "ngraph/file_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {
TensorExternalData::TensorExternalData(const ONNX_NAMESPACE::TensorProto& tensor) {
    for (const auto& entry : tensor.external_data()) {
        if (entry.key() == "location") {
            NGRAPH_SUPPRESS_DEPRECATED_START
            m_data_location = file_util::sanitize_path(entry.value());
            NGRAPH_SUPPRESS_DEPRECATED_END
        } else if (entry.key() == "offset") {
            m_offset = std::stoull(entry.value());
        } else if (entry.key() == "length") {
            m_data_length = std::stoull(entry.value());
        } else if (entry.key() == "checksum") {
            m_sha1_digest = entry.value();
        }
    }
    if (m_sha1_digest.size() > 0) {
        OPENVINO_WARN << "SHA1 checksum is not supported";
    }
}

Buffer<ov::MappedMemory> TensorExternalData::load_external_mmap_data(const std::string& model_dir) const {
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto full_path = file_util::path_join(model_dir, m_data_location);
    NGRAPH_SUPPRESS_DEPRECATED_END
    const int64_t file_size = ov::util::file_size(full_path);
    if (file_size <= 0 || m_offset + m_data_length > static_cast<uint64_t>(file_size)) {
        throw error::invalid_external_data{*this};
    }
    auto mapped_memory = ov::load_mmap_object(full_path, m_data_length, m_offset);
    if (m_data_length > mapped_memory->size() || mapped_memory->size() == 0) {
        throw error::invalid_external_data{*this};
    }
    return std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data(),
                                                                                              mapped_memory->size(),
                                                                                              mapped_memory);
}

Buffer<ngraph::runtime::AlignedBuffer> TensorExternalData::load_external_data(const std::string& model_dir) const {
    NGRAPH_SUPPRESS_DEPRECATED_START

    auto full_path = file_util::path_join(model_dir, m_data_location);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    file_util::convert_path_win_style(full_path);
    std::ifstream external_data_stream(ov::util::string_to_wstring(full_path).c_str(),
                                       std::ios::binary | std::ios::in | std::ios::ate);
#else
    std::ifstream external_data_stream(full_path, std::ios::binary | std::ios::in | std::ios::ate);
#endif
    NGRAPH_SUPPRESS_DEPRECATED_END

    if (external_data_stream.fail()) {
        throw error::invalid_external_data{*this};
    }
    const uint64_t file_size = static_cast<uint64_t>(external_data_stream.tellg());
    if (m_offset + m_data_length > file_size) {
        throw error::invalid_external_data{*this};
    }

    uint64_t read_data_length = m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset;

    // default value of m_offset is 0
    external_data_stream.seekg(m_offset, std::ios::beg);

    auto read_data = std::make_shared<ngraph::runtime::AlignedBuffer>(read_data_length);
    external_data_stream.read(read_data->get_ptr<char>(), read_data_length);
    external_data_stream.close();

    auto buffer = std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
        read_data->get_ptr<char>(),
        read_data->size(),
        read_data);

    return buffer;
}

std::string TensorExternalData::to_string() const {
    std::stringstream s;
    s << "ExternalDataInfo(";
    s << "data_full_path: " << m_data_location;
    s << ", offset: " << m_offset;
    s << ", data_length: " << m_data_length;
    if (m_sha1_digest.size() > 0) {
        s << ", sha1_digest: " << m_sha1_digest << ")";
    } else {
        s << ")";
    }
    return s.str();
}
}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
