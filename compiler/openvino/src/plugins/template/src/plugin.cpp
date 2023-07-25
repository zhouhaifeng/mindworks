// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <memory>

#include "ie_plugin_config.hpp"
#include "itt.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "remote_context.hpp"
#include "template/properties.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"

namespace {
static constexpr const char* wait_executor_name = "TemplateWaitExecutor";
static constexpr const char* stream_executor_name = "TemplateStreamsExecutor";
static constexpr const char* template_exclusive_executor = "TemplateExecutor";
}  // namespace

// ! [plugin:ctor]
ov::template_plugin::Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    set_device_name("TEMPLATE");

    // create ngraph backend which performs inference using ngraph reference implementations
    m_backend = ov::runtime::Backend::create();

    // create default stream executor with a given name
    m_waitExecutor = get_executor_manager()->get_idle_cpu_streams_executor({wait_executor_name});
}
// ! [plugin:ctor]

// ! [plugin:dtor]
ov::template_plugin::Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    get_executor_manager()->clear(stream_executor_name);
    get_executor_manager()->clear(wait_executor_name);
}
// ! [plugin:dtor]

// ! [plugin:create_context]
ov::SoPtr<ov::IRemoteContext> ov::template_plugin::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    return std::make_shared<ov::template_plugin::RemoteContext>();
}
// ! [plugin:create_context]

// ! [plugin:get_default_context]
ov::SoPtr<ov::IRemoteContext> ov::template_plugin::Plugin::get_default_context(
    const ov::AnyMap& remote_properties) const {
    return std::make_shared<ov::template_plugin::RemoteContext>();
}
// ! [plugin:get_default_context]

// ! [plugin:transform_model]
void transform_model(const std::shared_ptr<ov::Model>& model) {
    // Perform common optimizations and device-specific transformations
    ov::pass::Manager passManager;
    // Example: register CommonOptimizations transformation from transformations library
    passManager.register_pass<ov::pass::CommonOptimizations>();
    // Disable some transformations
    passManager.get_pass_config()->disable<ov::pass::UnrollIf>();
    // This transformation changes output name
    passManager.get_pass_config()->disable<ov::pass::ConvertReduceSumToPooling>();
    // Register any other transformations
    // ..

    const auto& pass_config = passManager.get_pass_config();

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // After `run_passes`, we have the transformed function, where operations match device operations,
    // and we can create device backend-dependent graph
    passManager.run_passes(model);
}
// ! [plugin:transform_model]

// ! [plugin:compile_model]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    return compile_model(model, properties, {});
}
// ! [plugin:compile_model]

// ! [plugin:compile_model_with_remote]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::compile_model");

    auto fullConfig = Configuration{properties, m_cfg};
    auto streamsExecutorConfig =
        ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(fullConfig.streams_executor_config);
    streamsExecutorConfig._name = stream_executor_name;
    auto compiled_model = std::make_shared<CompiledModel>(
        model->clone(),
        shared_from_this(),
        context,
        fullConfig.exclusive_async_requests
            ? get_executor_manager()->get_executor(template_exclusive_executor)
            : get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig),
        fullConfig);
    return compiled_model;
}
// ! [plugin:compile_model_with_remote]

// ! [plugin:import_model]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(std::istream& model,
                                                                              const ov::AnyMap& properties) const {
    return import_model(model, {}, properties);
}
// ! [plugin:import_model]

// ! [plugin:import_model_with_remote]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(
    std::istream& model,
    const ov::SoPtr<ov::IRemoteContext>& context,
    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::import_model");

    auto fullConfig = Configuration{properties, m_cfg};
    // read XML content
    std::string xmlString;
    std::uint64_t dataSize = 0;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    xmlString.resize(dataSize);
    model.read(const_cast<char*>(xmlString.c_str()), dataSize);

    // read blob content
    ov::Tensor weights;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    if (0 != dataSize) {
        weights = ov::Tensor(ov::element::from<char>(), ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
        model.read(weights.data<char>(), dataSize);
    }

    auto ov_model = get_core()->read_model(xmlString, weights);
    auto streamsExecutorConfig =
        ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(fullConfig.streams_executor_config);
    streamsExecutorConfig._name = stream_executor_name;
    auto compiled_model =
        std::make_shared<CompiledModel>(ov_model,
                                        shared_from_this(),
                                        context,
                                        get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig),
                                        fullConfig,
                                        true);
    return compiled_model;
}
// ! [plugin:import_model_with_remote]

// ! [plugin:query_model]
ov::SupportedOpsMap ov::template_plugin::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                             const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::query_model");

    Configuration fullConfig{properties, m_cfg, false};

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    auto supported = ov::get_supported_nodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            // skip transformations in case of user config
            if (fullConfig.disable_transformations)
                return;
            // 1. It is needed to apply all transformations as it is done in compile_model
            transform_model(model);
        },
        [&](std::shared_ptr<ngraph::Node> node) {
            // 2. Сheck whether node is supported
            ngraph::OpSet op_super_set;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) op_super_set.insert<NAMESPACE::NAME>();
        // clang-format off
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset11_tbl.hpp"
#include "openvino/opsets/opset12_tbl.hpp"
        // clang-format on
#undef _OPENVINO_OP_REG
            return op_super_set.contains_type(node->get_type_info());
        });

    // 3. Produce the result
    ov::SupportedOpsMap res;
    for (auto&& layerName : supported) {
        res.emplace(layerName, get_device_name() + "." + std::to_string(m_cfg.device_id));
    }

    return res;
}
// ! [plugin:query_model]

// ! [plugin:set_property]
void ov::template_plugin::Plugin::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}
// ! [plugin:set_property]

// ! [plugin:get_property]
ov::Any ov::template_plugin::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::available_devices,
                                                    ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::architecture,
                                                    ov::device::capabilities,
                                                    ov::range_for_async_infer_requests};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::id,
                                                    ov::enable_profiling,
                                                    ov::hint::performance_mode,
                                                    ov::internal::exclusive_async_requests,
                                                    ov::template_plugin::disable_transformations};
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();

        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        add_ro_properties(METRIC_KEY(IMPORT_EXPORT_SUPPORT), metrics);
        return to_string_vector(metrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        auto configs = default_rw_properties();
        auto streamExecutorConfigKeys = ov::threading::IStreamsExecutor::Config{}
                                            .get_property(ov::supported_properties.name())
                                            .as<std::vector<std::string>>();
        for (auto&& configKey : streamExecutorConfigKeys) {
            if (configKey != InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) {
                configs.emplace_back(configKey);
            }
        }
        return to_string_vector(configs);
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}};
    } else if (ov::available_devices == name) {
        // TODO: fill list of available devices
        std::vector<std::string> available_devices = {""};
        return decltype(ov::available_devices)::value_type(available_devices);
    } else if (ov::device::full_name == name) {
        std::string device_name = "Template Device Full Name";
        return decltype(ov::device::full_name)::value_type(device_name);
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        return true;
    } else if (ov::device::architecture == name) {
        // TODO: return device architecture for device specified by DEVICE_ID config
        std::string arch = "TEMPLATE";
        return decltype(ov::device::architecture)::value_type(arch);
    } else if (ov::internal::caching_properties == name) {
        std::vector<ov::PropertyName> caching_properties = {ov::device::architecture};
        return decltype(ov::internal::caching_properties)::value_type(caching_properties);
    } else if (ov::device::capabilities == name) {
        // TODO: fill actual list of supported capabilities: e.g. Template device supports only FP32 and EXPORT_IMPORT
        std::vector<std::string> capabilities = {ov::device::capability::FP32, ov::device::capability::EXPORT_IMPORT};
        return decltype(ov::device::capabilities)::value_type(capabilities);
    } else if (ov::range_for_async_infer_requests == name) {
        // TODO: fill with actual values
        using uint = unsigned int;
        return decltype(ov::range_for_async_infer_requests)::value_type(std::make_tuple(uint{1}, uint{1}, uint{1}));
    } else {
        return m_cfg.Get(name);
    }
}
// ! [plugin:get_property]

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_template_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::template_plugin::Plugin, version)
// ! [plugin:create_plugin_engine]
