// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <vector>

namespace cldnn {
const char *debug_configuration::prefix = "GPU_Debug: ";

// Default policy is that dump_configuration will override other configuration from IE.

#ifdef GPU_DEBUG_CONFIG

#define GPU_DEBUG_COUT_ std::cout << cldnn::debug_configuration::prefix

template<typename T>
void print_option(std::string option_name, T option_value) {
    GPU_DEBUG_COUT_ << "Config " << option_name << " = " << option_value << std::endl;
}

static std::string to_upper_case(const std::string& var) {
    std::stringstream s;

    for (size_t i = 0; i < var.size(); i++) {
        if (std::isupper(var[i])) {
            if (i != 0) {
                s << "_";
            }
            s << var[i];
        } else {
            s << static_cast<char>(std::toupper(var[i]));
        }
    }

    return s.str();
}

static std::vector<std::string> get_possible_option_names(const std::string& var, std::vector<std::string> allowed_option_prefixes) {
    std::vector<std::string> result;

    for (auto& prefix : allowed_option_prefixes) {
        result.push_back(prefix + var);
        result.push_back(prefix + to_upper_case(var));
    }

    return result;
}

template <typename T>
T convert_to(const std::string &str) {
    std::istringstream ss(str);
    T res;
    ss >> res;
    return res;
}

template <>
std::string convert_to(const std::string &str) {
    return str;
}

template<typename T>
void get_debug_env_var(const std::string &var, T &val, std::vector<std::string> allowed_option_prefixes) {
    bool found = false;
    for (auto o : get_possible_option_names(var, allowed_option_prefixes)) {
        if (const auto env_var = std::getenv(o.c_str())) {
            val = convert_to<T>(env_var);
            found = true;
        }
    }

    if (found) {
        print_option(var, val);
    }
}

template<typename T>
void get_gpu_debug_env_var(const std::string &var, T &val) {
    return get_debug_env_var(var, val, {"OV_GPU_"});
}

template<typename T>
void get_common_debug_env_var(const std::string &var, T &val) {
    // The list below should be prioritized from lowest to highest prefix priority
    // If an option is set several times with different prefixes, version with the highest priority will be actually used.
    // This may allow to enable global option with some value and override this value for GPU plugin
    // For example: OV_GPU_Verbose=2 OV_Verbose=1 ./my_app => this->verbose == 2
    // In that case we enable Verbose (with level = 1) for all OV components that support this option, but for GPU plugin we increase verbose level to 2
    std::vector<std::string> allowed_option_prefixes = {
        "OV_",
        "OV_GPU_"
    };

    return get_debug_env_var(var, val, allowed_option_prefixes);
}

static void print_help_messages() {
    std::vector<std::pair<std::string, std::string>> message_list;
    message_list.emplace_back("OV_GPU_Help", "Print help messages");
    message_list.emplace_back("OV_GPU_Verbose", "Verbose execution");
    message_list.emplace_back("OV_GPU_VerboseColor", "Print verbose color");
    message_list.emplace_back("OV_GPU_ListLayers", "Print layers names");
    message_list.emplace_back("OV_GPU_PrintMultiKernelPerf", "Print execution time of each kernel in multi-kernel primitimive");
    message_list.emplace_back("OV_GPU_DisableUsm", "Disable usm usage");
    message_list.emplace_back("OV_GPU_DisableOnednn", "Disable onednn for discrete GPU (no effect for integrated GPU)");
    message_list.emplace_back("OV_GPU_DisableOnednnOptPostOps", "Disable onednn optimize post operators");
    message_list.emplace_back("OV_GPU_DumpProfilingData", "Enables dump of extended profiling information to specified directory."
                              " Note: Performance impact may be significant as this option enforces host side sync after each primitive");
    message_list.emplace_back("OV_GPU_DumpGraphs", "Dump optimized graph");
    message_list.emplace_back("OV_GPU_DumpSources", "Dump opencl sources");
    message_list.emplace_back("OV_GPU_DumpLayersPath", "Enable dumping intermediate buffers and set the dest path");
    message_list.emplace_back("OV_GPU_DumpLayers", "Dump intermediate buffers of specified layers only, separated by space."
                               " Support case-insensitive and regular expression. For example .*conv.*");
    message_list.emplace_back("OV_GPU_DumpLayersResult", "Dump output buffers of result layers only");
    message_list.emplace_back("OV_GPU_DumpLayersDstOnly", "Dump only output of layers");
    message_list.emplace_back("OV_GPU_DumpLayersLimitBatch", "Limit the size of batch to dump");
    message_list.emplace_back("OV_GPU_DumpLayersRaw", "If true, dump data is stored in raw memory format.");
    message_list.emplace_back("OV_GPU_DryRunPath", "Dry run and serialize execution graph into the specified path");
    message_list.emplace_back("OV_GPU_BaseBatchForMemEstimation", "Base batch size to be used in memory estimation");
    message_list.emplace_back("OV_GPU_AfterProc", "Run inference after the specified process PIDs are finished, separated by space."
                              " Supported on only on linux.");
    message_list.emplace_back("OV_GPU_SerialCompile", "Serialize creating primitives and compiling kernels");
    message_list.emplace_back("OV_GPU_ForceImplTypes", "Force implementation type of a target primitive or layer. [primitive or layout_name]:[impl_type]"
                              " For example fc:onednn gemm:onednn reduce:ocl do:cpu"
                              " For primitives fc, gemm, do, reduce, concat are supported. Separated by space.");
    message_list.emplace_back("OV_GPU_MaxKernelsPerBatch", "Maximum number of kernels in a batch during compiling kernels");
    message_list.emplace_back("OV_GPU_DisableAsyncCompilation", "Disable async compilation");
    message_list.emplace_back("OV_GPU_DisableDynamicImpl", "Disable dynamic implementation");
    message_list.emplace_back("OV_GPU_DumpIteration", "Dump n-th execution of network, separated by space.");
    message_list.emplace_back("OV_GPU_MemPreallocationOptions", "Controls buffer pre-allocation feature. Expects 4 values separated by space in"
                              "the following order: number of iterations for pre-allocation(int), max size of single iteration in bytes(int), "
                              "max per-dim allowed diff(int), unconditional buffers preallocation ratio(float). For example for disabling memory"
                              "preallocation at all, you can use OV_GPU_MemPreallocationOptions='0 0 0 1.0'");

    auto max_name_length_item = std::max_element(message_list.begin(), message_list.end(),
        [](std::pair<std::string, std::string>& a, std::pair<std::string, std::string>& b){
            return a.first.size() < b.first.size();
    });
    int name_width = static_cast<int>(max_name_length_item->first.size()) + 2;

    GPU_DEBUG_COUT_ << "Supported environment variables for debugging" << std::endl;
    for (auto& p : message_list) {
        GPU_DEBUG_COUT_ << " - " << std::left << std::setw(name_width) << p.first + "  " << p.second << std::endl;
    }
}

#endif

debug_configuration::debug_configuration()
        : help(0)
        , verbose(0)
        , verbose_color(0)
        , list_layers(0)
        , print_multi_kernel_perf(0)
        , disable_usm(0)
        , disable_onednn(0)
        , disable_onednn_opt_post_ops(0)
        , dump_profiling_data(std::string(""))
        , dump_graphs(std::string())
        , dump_sources(std::string())
        , dump_layers_path(std::string())
        , dry_run_path(std::string())
        , dump_layers_dst_only(0)
        , dump_layers_result(0)
        , dump_layers_limit_batch(std::numeric_limits<int>::max())
        , dump_layers_raw(0)
        , base_batch_for_memory_estimation(-1)
        , serialize_compile(0)
        , max_kernels_per_batch(0)
        , disable_async_compilation(0)
        , disable_dynamic_impl(0) {
#ifdef GPU_DEBUG_CONFIG
    get_gpu_debug_env_var("Help", help);
    get_common_debug_env_var("Verbose", verbose);
    get_gpu_debug_env_var("VerboseColor", verbose_color);
    get_gpu_debug_env_var("ListLayers", list_layers);
    get_gpu_debug_env_var("PrintMultiKernelPerf", print_multi_kernel_perf);
    get_gpu_debug_env_var("DisableUsm", disable_usm);
    get_gpu_debug_env_var("DumpGraphs", dump_graphs);
    get_gpu_debug_env_var("DumpSources", dump_sources);
    get_gpu_debug_env_var("DumpLayersPath", dump_layers_path);
    get_gpu_debug_env_var("DumpLayersLimitBatch", dump_layers_limit_batch);
    get_gpu_debug_env_var("DumpLayersRaw", dump_layers_raw);
    get_gpu_debug_env_var("DumpLayersDstOnly", dump_layers_dst_only);
    get_gpu_debug_env_var("DumpLayersResult", dump_layers_result);
    get_gpu_debug_env_var("DisableOnednn", disable_onednn);
    get_gpu_debug_env_var("DisableOnednnOptPostOps", disable_onednn_opt_post_ops);
    get_gpu_debug_env_var("DumpProfilingData", dump_profiling_data);
    get_gpu_debug_env_var("DryRunPath", dry_run_path);
    get_gpu_debug_env_var("BaseBatchForMemEstimation", base_batch_for_memory_estimation);
    std::string dump_layers_str;
    get_gpu_debug_env_var("DumpLayers", dump_layers_str);
    std::string after_proc_str;
    get_gpu_debug_env_var("AfterProc", after_proc_str);
    get_gpu_debug_env_var("SerialCompile", serialize_compile);
    std::string forced_impl_types_str;
    get_gpu_debug_env_var("ForceImplTypes", forced_impl_types_str);
    get_gpu_debug_env_var("MaxKernelsPerBatch", max_kernels_per_batch);
    get_gpu_debug_env_var("DisableAsyncCompilation", disable_async_compilation);
    get_gpu_debug_env_var("DisableDynamicImpl", disable_dynamic_impl);
    std::string dump_iteration_str;
    get_gpu_debug_env_var("DumpIteration", dump_iteration_str);
    std::string mem_preallocation_params_str;
    get_gpu_debug_env_var("MemPreallocationOptions", mem_preallocation_params_str);

    if (help > 0) {
        print_help_messages();
        exit(0);
    }

    if (dump_layers_str.length() > 0) {
        dump_layers_str = " " + dump_layers_str + " "; // Insert delimiter for easier parsing when used
        std::stringstream ss(dump_layers_str);
        std::string layer;
        while (ss >> layer) {
            dump_layers.push_back(layer);
        }
    }

    if (forced_impl_types_str.length() > 0) {
        forced_impl_types_str = " " + forced_impl_types_str + " "; // Insert delimiter for easier parsing when used
        std::stringstream ss(forced_impl_types_str);
        std::string type;
        while (ss >> type) {
            forced_impl_types.push_back(type);
        }
    }

    if (dump_iteration_str.size() > 0) {
        dump_iteration_str = " " + dump_iteration_str + " ";
        std::istringstream ss(dump_iteration_str);
        std::string token;
        while (ss >> token) {
            try {
                dump_iteration.insert(static_cast<int64_t>(std::stol(token)));
            } catch(const std::exception& ex) {
                dump_iteration.clear();
                GPU_DEBUG_COUT << "OV_GPU_DumpIteration was ignored. It cannot be parsed to integer array." << std::endl;
                break;
            }
        }
    }

    if (mem_preallocation_params_str.size() > 0) {
        mem_preallocation_params_str = " " + mem_preallocation_params_str + " ";
        std::istringstream ss(mem_preallocation_params_str);
        std::vector<std::string> params;
        std::string param;
        while (ss >> param)
            params.push_back(param);

        bool correct_params = params.size() == 4;
        if (correct_params) {
            try {
                mem_preallocation_params.next_iters_preallocation_count = std::stol(params[0]);
                mem_preallocation_params.max_per_iter_size = std::stol(params[1]);
                mem_preallocation_params.max_per_dim_diff = std::stol(params[2]);
                mem_preallocation_params.buffers_preallocation_ratio = std::stof(params[3]);
            } catch(const std::exception& ex) {
                correct_params = false;
            }
        }

        if (!correct_params)
            GPU_DEBUG_COUT_ << "OV_GPU_MemPreallocationOptions were ignored, because they cannot be parsed.\n";

        mem_preallocation_params.is_initialized = correct_params;
    }

    if (after_proc_str.length() > 0) {
#ifdef _WIN32
        GPU_DEBUG_COUT_ << "Warning: OV_GPU_AfterProc is supported only on linux" << std::endl;
#else
        after_proc_str = " " + after_proc_str + " "; // Insert delimiter for easier parsing when used
        std::stringstream ss(after_proc_str);
        std::string pid;
        while (ss >> pid) {
            after_proc.push_back(pid);
        }
#endif
    }
#endif
}

const debug_configuration *debug_configuration::get_instance() {
    static std::unique_ptr<debug_configuration> instance(nullptr);
#ifdef GPU_DEBUG_CONFIG
    static std::mutex _m;
    std::lock_guard<std::mutex> lock(_m);
    if (nullptr == instance)
        instance.reset(new debug_configuration());
    return instance.get();
#else
    return nullptr;
#endif
}

bool debug_configuration::is_dumped_layer(const std::string& layer_name, bool is_output) const {
#ifdef GPU_DEBUG_CONFIG
    if (is_output == true && dump_layers_result == 1 &&
        (layer_name.find("constant:") == std::string::npos))
        return true;
    if (dump_layers.empty() && dump_layers_result == 0)
        return true;

    auto is_match = [](const std::string& layer_name, const std::string& pattern) -> bool {
        auto upper_layer_name = std::string(layer_name.length(), '\0');
        std::transform(layer_name.begin(), layer_name.end(), upper_layer_name.begin(), ::toupper);
        auto upper_pattern = std::string(pattern.length(), '\0');
        std::transform(pattern.begin(), pattern.end(), upper_pattern.begin(), ::toupper);
        // Check pattern from exec_graph
        size_t pos = upper_layer_name.find(':');
        auto upper_exec_graph_name = upper_layer_name.substr(pos + 1, upper_layer_name.size());
        if (upper_exec_graph_name.compare(upper_pattern) == 0) {
            return true;
        }
        // Check pattern with regular expression
        std::regex re(upper_pattern);
        return std::regex_match(upper_layer_name, re);
    };
    auto iter = std::find_if(dump_layers.begin(), dump_layers.end(), [&](const std::string& dl){
        return is_match(layer_name, dl);
    });
    return (iter != dump_layers.end());
#else
    return false;
#endif
}

bool debug_configuration::is_target_iteration(int64_t iteration) const {
#ifdef GPU_DEBUG_CONFIG
    if (iteration < 0)
        return true;

    if (dump_iteration.empty())
        return true;

    if (dump_iteration.find(iteration) == std::end(dump_iteration))
        return false;

    return true;
#else
    return false;
#endif
}
} // namespace cldnn
