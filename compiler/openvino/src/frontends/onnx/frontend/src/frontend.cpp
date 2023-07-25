// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION >= 4022000  // protobuf 4.22
#    define OV_PROTOBUF_ABSL_IS_USED
#endif
#include <google/protobuf/port_undef.inc>

#ifndef OV_PROTOBUF_ABSL_IS_USED
#    include <google/protobuf/stubs/logging.h>
#endif

#include <fstream>
#include <input_model.hpp>
#include <onnx_import/onnx.hpp>
#include <onnx_import/onnx_utils.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/frontend/onnx/extension/conversion.hpp>
#include <openvino/frontend/onnx/frontend.hpp>
#include <openvino/frontend/onnx/visibility.hpp>
#include <sstream>
#include <utils/onnx_internal.hpp>

#include "legacy_op_extension.hpp"
#include "onnx_common/onnx_model_validator.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "ops_bridge.hpp"
#include "utils/common.hpp"

using namespace ov;
using namespace ov::frontend::onnx;

ONNX_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

ONNX_FRONTEND_C_API void* get_front_end_data() {
    ov::frontend::FrontEndPluginInfo* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "onnx";
    res->m_creator = []() {
        return std::make_shared<FrontEnd>();
    };
#ifndef OPENVINO_DEBUG_ENABLE
    // disable protobuf logging
#    ifndef OV_PROTOBUF_ABSL_IS_USED
    google::protobuf::SetLogHandler(nullptr);
#    endif
#endif
    return res;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.empty()) {
        return nullptr;
    }
    // enable mmap by default
    const bool enable_mmap = variants[variants.size() - 1].is<bool>() ? variants[variants.size() - 1].as<bool>() : true;

    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        return std::make_shared<InputModel>(path, enable_mmap, m_extensions);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        return std::make_shared<InputModel>(path, enable_mmap, m_extensions);
    }
#endif
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        if (variants.size() > 1 && variants[1].is<std::string>()) {
            const auto path = variants[1].as<std::string>();
            return std::make_shared<InputModel>(*stream, path, enable_mmap, m_extensions);
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        if (variants.size() > 1 && variants[1].is<std::wstring>()) {
            const auto path = variants[1].as<std::wstring>();
            return std::make_shared<InputModel>(*stream, path, enable_mmap, m_extensions);
        }
#endif
        return std::make_shared<InputModel>(*stream, enable_mmap, m_extensions);
    }
    return nullptr;
}

std::shared_ptr<ngraph::Function> FrontEnd::convert_partially(const InputModel::Ptr& model) const {
    auto model_onnx = std::dynamic_pointer_cast<InputModel>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto function = decode(model);

        ov::pass::Manager manager;
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(function);
        convert(function);
        return function;
    }

    return model_onnx->convert();
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    // Here, you can register transformations as a second step of importing process
    // In particular, you can operate on not supported ops (it allows to N:N ONNX->OV mapping).
    // ov::pass::Manager manager;
    // manager.run_passes(model);
}

std::shared_ptr<ngraph::Function> FrontEnd::convert(const InputModel::Ptr& model) const {
    const auto partially_converted = convert_partially(model);

    const auto error_message = ngraph::onnx_import::common::collect_translation_exceptions(partially_converted);
    NGRAPH_CHECK(error_message.empty(), error_message);

    normalize(partially_converted);

    return partially_converted;
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partially_converted) const {
    ngraph::onnx_import::detail::convert_decoded_function(partially_converted);
}

std::shared_ptr<ngraph::Function> FrontEnd::decode(const InputModel::Ptr& model) const {
    auto model_onnx = std::dynamic_pointer_cast<InputModel>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->decode();
}

std::string FrontEnd::get_name() const {
    return "onnx";
}

namespace {
/**
 * This helper struct uses RAII to rewind/reset the stream so that it points to the beginning
 * of the underlying resource (string, file, and so on). It works similarly to std::lock_guard,
 * which releases a mutex upon destruction.
 *
 * This ensures that the stream is always reset (exception, successful and unsuccessful
 * model validation).
 */
struct StreamRewinder {
    StreamRewinder(std::istream& stream) : m_stream(stream) {
        m_stream.seekg(0, m_stream.beg);
    }
    ~StreamRewinder() {
        m_stream.seekg(0, m_stream.beg);
    }

private:
    std::istream& m_stream;
};
}  // namespace

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 0) {
        return false;
    }
    std::ifstream model_stream;
    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        model_stream.open(path.c_str(), std::ios::in | std::ifstream::binary);
    }
#endif
    if (model_stream.is_open()) {
        model_stream.seekg(0, model_stream.beg);
        const bool is_valid_model = ngraph::onnx_common::is_valid_model(model_stream);
        model_stream.close();
        return is_valid_model;
    }
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        StreamRewinder rwd{*stream};
        return ngraph::onnx_common::is_valid_model(*stream);
    }

    return false;
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_extensions.telemetry = telemetry;
    } else if (auto transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
        m_transformation_extensions.push_back(transformation);
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_other_extensions.push_back(so_ext);
    } else if (auto common_conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtension>(extension)) {
        m_extensions.conversions.push_back(common_conv_ext);
    } else if (const auto onnx_conv_ext = std::dynamic_pointer_cast<onnx::ConversionExtension>(extension)) {
        m_extensions.conversions.push_back(onnx_conv_ext);
    } else if (auto progress_reporter = std::dynamic_pointer_cast<ProgressReporterExtension>(extension)) {
        m_extensions.progress_reporter = progress_reporter;
    } else if (const auto& legacy_ext = std::dynamic_pointer_cast<ov::LegacyOpExtension>(extension)) {
        m_other_extensions.push_back(legacy_ext);
        std::call_once(has_legacy_extension, [this] {
            m_extensions.conversions.push_back(ngraph::onnx_import::detail::get_legacy_conversion_extension());
        });
    }
}
