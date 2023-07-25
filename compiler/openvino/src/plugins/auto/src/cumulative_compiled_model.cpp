// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cumulative_compiled_model.hpp"
#include "common.hpp"
#include <memory>

#include "async_infer_request.hpp"
#include "itt.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin.hpp"
#include "ie_plugin_config.hpp"

namespace ov {
namespace auto_plugin {
AutoCumuCompiledModel::AutoCumuCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::SoPtr<ov::IRemoteContext>& remote_context,
                                             ScheduleContext::Ptr& schedule_context,
                                             Schedule::Ptr& scheduler)
    : CompiledModel(model, plugin, remote_context, schedule_context, scheduler) {
      m_scheduler = std::dynamic_pointer_cast<CumuSchedule>(scheduler);
}

void AutoCumuCompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> AutoCumuCompiledModel::get_runtime_model() const {
    if (m_context->m_hw_compiled_model)
        return m_context->m_hw_compiled_model->get_runtime_model();
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any AutoCumuCompiledModel::get_property(const std::string& name) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::supported_properties,
                                                    ov::execution_devices,
                                                    ov::hint::performance_mode,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::device::properties,
                                                    ov::hint::model_priority,
                                                    ov::loaded_from_cache};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities};
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    if (name == ov::supported_properties) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (name == ov::hint::performance_mode) {
        return m_context->m_performance_hint;
    } else if (name == ov::device::priorities) {
        // device priority does not support change on-the-fly
        return decltype(ov::device::priorities)::value_type(m_context->m_str_devices);
    } else if (name == ov::device::properties) {
        ov::AnyMap all_devices = {};
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                auto temp = get_device_supported_properties(m_scheduler->m_p_ctput_loadcontext[i]);
                all_devices.insert(temp.begin(), temp.end());
            }
        }
        return all_devices;
    } else if (name == ov::hint::model_priority) {
        auto value = m_context->m_model_priority;
        if (m_context->m_ov_core->is_new_api()) {
            return value ? ((value > 1) ? ov::hint::Priority::LOW :
                    ov::hint::Priority::MEDIUM) : ov::hint::Priority::HIGH;
        } else {
            OPENVINO_SUPPRESS_DEPRECATED_START
            return value ? ((value > 1) ? CONFIG_VALUE(MODEL_PRIORITY_LOW) : CONFIG_VALUE(
                        MODEL_PRIORITY_MED)) : CONFIG_VALUE(MODEL_PRIORITY_HIGH);
            OPENVINO_SUPPRESS_DEPRECATED_END
        }
    } else if (name == ov::optimal_number_of_infer_requests) {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        unsigned int res = 0u;
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            try {
                if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                    res += (m_scheduler->m_p_ctput_loadcontext[i])
                                .m_compiled_model->get_property(ov::optimal_number_of_infer_requests.name())
                                .as<unsigned int>();
                }
            } catch (const ov::Exception& err) {
                OPENVINO_THROW("Every device used in cumulative mode should support OPTIMAL_NUMBER_OF_INFER_REQUESTS property from compiled model",
                        "Failed to query the property with error:", err.what());
            }
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type {res};
    } else if (name == ov::execution_devices) {
        std::vector<std::string> exeDevices = {};
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        for (auto const & n : m_context->m_device_priorities) {
            exeDevices.push_back(n.device_name);
        }
        return decltype(ov::execution_devices)::value_type {exeDevices};
    } else if (name == ov::model_name) {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                return m_scheduler->m_p_ctput_loadcontext[i].m_compiled_model->get_property(name);
            }
        }
        OPENVINO_THROW("No valid compiled model found to get", name);
    OPENVINO_SUPPRESS_DEPRECATED_START
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        auto ro_properties = default_ro_properties();
        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), ro_properties);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), ro_properties);
        return to_string_vector(ro_properties);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        auto rw_properties = default_rw_properties();
        return to_string_vector(rw_properties);
    OPENVINO_SUPPRESS_DEPRECATED_END
    } else if (name == ov::loaded_from_cache) {
        bool loaded_from_cache = true;
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                loaded_from_cache &= (m_scheduler->m_p_ctput_loadcontext[i].m_compiled_model->get_property(name).as<bool>());
            }
        }
        return loaded_from_cache;
    }
    OPENVINO_THROW(get_log_tag(), ": not supported property ", name);;
}

void AutoCumuCompiledModel::export_model(std::ostream& model_stream) const {
    OPENVINO_NOT_IMPLEMENTED;
}
} // namespace auto_plugin
} // namespace ov
