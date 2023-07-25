// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <legacy/ie_util_internal.hpp>
#include <map>
#include <memory>
#include <string>

#include "gna_executable_network.hpp"
#include "gna_plugin_config.hpp"

namespace ov {
namespace intel_gna {

class GNAPluginInternal : public InferenceEngine::IInferencePlugin {
private:
    mutable std::mutex syncCalls;
    Config defaultConfig;
    std::weak_ptr<GNAPlugin> plgPtr;
    std::shared_ptr<GNAPlugin> GetCurrentPlugin() const {
        std::lock_guard<std::mutex> lock{syncCalls};
        auto ptr = plgPtr.lock();
        if (ptr == nullptr) {
            return std::make_shared<GNAPlugin>();
        } else {
            return ptr;
        }
    }

protected:
    std::string _pluginInternalName = "GNA";

public:
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override {
        std::lock_guard<std::mutex> lock{syncCalls};
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.keyConfigMap);
        plgPtr = plg;
        InferenceEngine::CNNNetwork clonedNetwork(InferenceEngine::cloneNetwork(network));
        return std::make_shared<GNAExecutableNetwork>(clonedNetwork, plg);
    }

    void SetConfig(const std::map<std::string, std::string>& config) override {
        defaultConfig.UpdateFromMap(config);
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) override {
        std::lock_guard<std::mutex> lock{syncCalls};
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.keyConfigMap);
        plgPtr = plg;
        auto network_impl = std::make_shared<GNAExecutableNetwork>(modelFileName, plg);
        // set pointer for IInferencePlugin interface
        network_impl->SetPointerToPlugin(shared_from_this());

        return network_impl;
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        std::lock_guard<std::mutex> lock{syncCalls};
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.keyConfigMap);
        plgPtr = plg;
        auto network_impl = std::make_shared<GNAExecutableNetwork>(networkModel, plg);
        // set pointer for IInferencePlugin interface
        network_impl->SetPointerToPlugin(shared_from_this());

        return network_impl;
    }

    std::string GetName() const noexcept override {
        auto ptr = plgPtr.lock();
        if (ptr == nullptr) {
            return _pluginInternalName;
        } else {
            return ptr->GetName();
        }
    }

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override {
        auto plg = GetCurrentPlugin();
        try {
            plg->SetConfig(config);
        } catch (InferenceEngine::Exception&) {
        }
        return plg->QueryNetwork(network, config);
    }

    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        return GetCurrentPlugin()->GetMetric(name, options);
    }

    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        return defaultConfig.GetParameter(name);
    }
};

}  // namespace intel_gna
}  // namespace ov
