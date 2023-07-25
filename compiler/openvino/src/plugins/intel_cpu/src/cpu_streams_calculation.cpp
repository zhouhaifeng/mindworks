// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <transformations/utils/utils.hpp>
#include <unordered_set>

#include "cpu_map_scheduling.hpp"
#include "graph.h"
#include "ie_system_conf.h"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "performance_heuristics.hpp"

using namespace ov;
using namespace threading;

#define INIT_VAL -100

namespace ov {
namespace intel_cpu {

std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const bool input_streams_changed,
                                                     const int input_threads,
                                                     const int input_infer_requests,
                                                     const int model_prefer_threads,
                                                     const std::string input_perf_hint,
                                                     const Config::LatencyThreadingMode latencyThreadingMode,
                                                     const std::vector<std::vector<int>> proc_type_table) {
    std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE, INIT_VAL);
    std::vector<std::vector<int>> streams_info_table;
    std::vector<std::vector<int>> proc_socket_table;

    int n_streams = 0;
    int n_threads_per_stream = 0;

    auto update_mix_stream_info = [&]() {
        stream_info[NUMBER_OF_STREAMS] = 0;
        int n_threads = stream_info[THREADS_PER_STREAM];
        for (int n = MAIN_CORE_PROC; n <= HYPER_THREADING_PROC; n++) {
            if (0 != proc_type_table[0][n]) {
                stream_info[PROC_TYPE] = n;
                if (n_threads <= proc_type_table[0][n]) {
                    stream_info[THREADS_PER_STREAM] = n_threads;
                    streams_info_table.push_back(stream_info);
                    break;
                } else {
                    stream_info[THREADS_PER_STREAM] = proc_type_table[0][n];
                    streams_info_table.push_back(stream_info);
                    n_threads -= proc_type_table[0][n];
                }
            }
        }
    };

    auto update_ids_method = [&](const std::vector<int>& one_proc_info) {
        stream_info[STREAM_NUMA_NODE_ID] = one_proc_info[PROC_NUMA_NODE_ID];
        stream_info[STREAM_SOCKET_ID] = one_proc_info[PROC_SOCKET_ID];
    };

    auto update_streams_per_node = [&](const int& proc_type, const std::vector<int>& one_proc_info) {
        if (0 != one_proc_info[proc_type]) {
            if (n_threads_per_stream == -1) {
                stream_info[THREADS_PER_STREAM] = (proc_type == EFFICIENT_CORE_PROC) ? 2 : 1;
            }
            stream_info[PROC_TYPE] = proc_type;
            update_ids_method(one_proc_info);
            stream_info[NUMBER_OF_STREAMS] =
                static_cast<int>(one_proc_info[proc_type] / stream_info[THREADS_PER_STREAM]);
            if ((stream_info[NUMBER_OF_STREAMS] == 0) && (proc_type == MAIN_CORE_PROC)) {
                stream_info[NUMBER_OF_STREAMS] =
                    static_cast<int>((one_proc_info[MAIN_CORE_PROC] + one_proc_info[HYPER_THREADING_PROC]) /
                                     stream_info[THREADS_PER_STREAM]);
            }
            if (n_streams < stream_info[NUMBER_OF_STREAMS]) {
                stream_info[NUMBER_OF_STREAMS] = n_streams;
            }
            streams_info_table.push_back(stream_info);

            n_streams -= stream_info[NUMBER_OF_STREAMS];
            proc_socket_table[one_proc_info[PROC_SOCKET_ID]][proc_type] -=
                stream_info[NUMBER_OF_STREAMS] * stream_info[THREADS_PER_STREAM];
        }
    };

    if (proc_type_table.size() == 1) {
        proc_socket_table.push_back(proc_type_table[0]);
    } else {
        std::unordered_set<int> socket_id_list(proc_type_table.size());
        for (size_t i = 1; i < proc_type_table.size(); i++) {
            if (!socket_id_list.count(proc_type_table[i][PROC_SOCKET_ID])) {
                proc_socket_table.push_back(proc_type_table[i]);
                socket_id_list.insert(proc_type_table[i][PROC_SOCKET_ID]);
            } else {
                for (auto& row : proc_socket_table) {
                    if (row[PROC_SOCKET_ID] == proc_type_table[i][PROC_SOCKET_ID]) {
                        for (int n = 0; n <= HYPER_THREADING_PROC; n++) {
                            row[n] += proc_type_table[i][n];
                        }
                        if (row[PROC_NUMA_NODE_ID] != proc_type_table[i][PROC_NUMA_NODE_ID]) {
                            row[PROC_NUMA_NODE_ID] = -1;
                        }
                    }
                }
            }
        }
    }

    if (((input_streams_changed == false) && (input_perf_hint == CONFIG_VALUE(LATENCY)) &&
         ((latencyThreadingMode == Config::LatencyThreadingMode::PER_PLATFORM) || (proc_type_table.size() == 1))) ||
        ((input_streams_changed == true) && (input_streams == 1))) {
        stream_info[NUMBER_OF_STREAMS] = 1;
        if (input_threads > 0) {
            stream_info[THREADS_PER_STREAM] = std::min(proc_type_table[0][ALL_PROC], input_threads);
            if ((stream_info[THREADS_PER_STREAM] > proc_type_table[0][MAIN_CORE_PROC]) &&
                (proc_type_table[0][MAIN_CORE_PROC] > 0) && (proc_type_table[0][EFFICIENT_CORE_PROC] > 0)) {
                stream_info[PROC_TYPE] = ALL_PROC;
                update_ids_method(proc_type_table[0]);
                streams_info_table.push_back(stream_info);
                update_mix_stream_info();
            } else {
                if ((stream_info[THREADS_PER_STREAM] <= proc_type_table[0][MAIN_CORE_PROC]) ||
                    (proc_type_table[0][EFFICIENT_CORE_PROC] == 0)) {
                    stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                } else {
                    stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
                }
                if (proc_type_table.size() == 1) {
                    update_ids_method(proc_type_table[0]);
                } else {
                    size_t i = 0;
                    for (i = 1; i < proc_type_table.size(); i++) {
                        if (proc_type_table[i][stream_info[PROC_TYPE]] >= stream_info[THREADS_PER_STREAM]) {
                            update_ids_method(proc_type_table[i]);
                            i = proc_type_table.size() + 1;
                            break;
                        }
                    }
                    if (i <= proc_type_table.size()) {
                        for (i = 0; i < proc_socket_table.size(); i++) {
                            if (proc_socket_table[i][stream_info[PROC_TYPE]] >= stream_info[THREADS_PER_STREAM]) {
                                update_ids_method(proc_socket_table[i]);
                                i = proc_socket_table.size() + 1;
                                break;
                            }
                        }
                        if (i <= proc_socket_table.size()) {
                            update_ids_method(proc_type_table[0]);
                        }
                    }
                }
                streams_info_table.push_back(stream_info);
            }
        } else {
            if (proc_type_table[0][ALL_PROC] == proc_type_table[0][EFFICIENT_CORE_PROC]) {
                stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
                stream_info[THREADS_PER_STREAM] =
                    (model_prefer_threads == 0)
                        ? proc_type_table[0][EFFICIENT_CORE_PROC]
                        : std::min(proc_type_table[0][EFFICIENT_CORE_PROC], model_prefer_threads);
                update_ids_method(proc_type_table[0]);
                streams_info_table.push_back(stream_info);
            } else if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
                       ((model_prefer_threads == 0) || (model_prefer_threads > proc_type_table[0][MAIN_CORE_PROC]))) {
                stream_info[PROC_TYPE] = ALL_PROC;
                stream_info[THREADS_PER_STREAM] =
                    (model_prefer_threads == 0 || model_prefer_threads > proc_type_table[0][MAIN_CORE_PROC])
                        ? proc_type_table[0][ALL_PROC]
                        : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
                update_ids_method(proc_type_table[0]);
                streams_info_table.push_back(stream_info);
                update_mix_stream_info();
            } else {
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                stream_info[THREADS_PER_STREAM] =
                    proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
                update_ids_method(proc_type_table[0]);
                streams_info_table.push_back(stream_info);
            }
        }
        return streams_info_table;
    } else if ((input_streams_changed == false) && (input_perf_hint == CONFIG_VALUE(LATENCY))) {
        if (latencyThreadingMode == Config::LatencyThreadingMode::PER_NUMA_NODE) {
            if (proc_type_table.size() == 1) {
                stream_info[NUMBER_OF_STREAMS] = 1;
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                stream_info[THREADS_PER_STREAM] = proc_type_table[0][ALL_PROC];
                update_ids_method(proc_type_table[0]);
                streams_info_table.push_back(stream_info);
            } else {
                for (size_t i = 1; i < proc_type_table.size(); i++) {
                    if (i != 1) {
                        if (proc_type_table[i][ALL_PROC] < streams_info_table[0][THREADS_PER_STREAM]) {
                            continue;
                        } else if (proc_type_table[i][ALL_PROC] < streams_info_table[0][THREADS_PER_STREAM]) {
                            streams_info_table.clear();
                        }
                    }
                    stream_info[NUMBER_OF_STREAMS] = 1;
                    stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                    stream_info[THREADS_PER_STREAM] = proc_type_table[i][ALL_PROC];
                    update_ids_method(proc_type_table[i]);
                    streams_info_table.push_back(stream_info);
                }
            }
        } else {
            for (size_t i = 0; i < proc_socket_table.size(); i++) {
                if (streams_info_table.size() != 0) {
                    if (streams_info_table[0][THREADS_PER_STREAM] > proc_socket_table[i][ALL_PROC]) {
                        continue;
                    } else if (streams_info_table[0][THREADS_PER_STREAM] < proc_socket_table[i][ALL_PROC]) {
                        streams_info_table.clear();
                    }
                }
                stream_info[NUMBER_OF_STREAMS] = 1;
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                stream_info[THREADS_PER_STREAM] = proc_socket_table[i][ALL_PROC];
                update_ids_method(proc_socket_table[i]);
                streams_info_table.push_back(stream_info);
            }
        }
        return streams_info_table;
    } else {
        int n_threads = 0;
        int base_type = MAIN_CORE_PROC;

        n_threads =
            (0 == input_threads) ? proc_type_table[0][ALL_PROC] : std::min(proc_type_table[0][ALL_PROC], input_threads);

        if ((input_streams_changed == true) && (input_streams > 0)) {
            base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
            n_streams = (input_infer_requests > 0) ? std::min(input_streams, input_infer_requests) : input_streams;
            if (n_streams >= n_threads) {
                n_streams = n_threads;
                n_threads_per_stream = 1;
            } else {
                n_threads_per_stream = std::min(std::max(1, n_threads / n_streams), proc_type_table[0][base_type]);
                if (proc_type_table.size() == 1) {
                    if ((n_threads_per_stream > proc_type_table[0][base_type]) &&
                        (n_threads_per_stream < proc_type_table[0][base_type] * 2)) {
                        n_threads_per_stream = proc_type_table[0][base_type];
                    } else if (n_threads_per_stream < proc_type_table[0][base_type]) {
                        n_threads_per_stream = static_cast<int>(
                            proc_type_table[0][base_type] /
                            ((proc_type_table[0][base_type] + n_threads_per_stream - 1) / n_threads_per_stream));
                    }
                }
            }
        } else {
            base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
            if (0 == model_prefer_threads) {
                int n_proc = (proc_type_table.size() == 1) ? std::min(n_threads, proc_type_table[0][base_type])
                                                           : std::min(n_threads, proc_type_table[1][base_type]);
                if (0 == n_proc % 4) {
                    n_threads_per_stream = 4;
                } else if (0 == n_proc % 5) {
                    n_threads_per_stream = 5;
                } else if (0 == n_proc % 3) {
                    n_threads_per_stream = 3;
                } else if (proc_type_table.size() == 1) {
                    n_threads_per_stream = n_proc;
                } else {
                    n_threads_per_stream = (n_proc > 16) ? 4 : std::max(1, static_cast<int>(n_proc / 4));
                }
                n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                if ((input_infer_requests > 0) && (n_streams > input_infer_requests)) {
                    n_streams = input_infer_requests;
                    n_threads_per_stream =
                        std::min(static_cast<int>(n_threads / n_streams), proc_type_table[0][base_type]);
                } else {
                    while (n_streams * 2 <= n_threads_per_stream) {
                        n_threads_per_stream = static_cast<int>(n_threads_per_stream / 2);
                        n_threads_per_stream = static_cast<int>(
                            proc_type_table[0][base_type] /
                            ((proc_type_table[0][base_type] + n_threads_per_stream - 1) / n_threads_per_stream));
                        n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                    }
                }
            } else if ((1 == model_prefer_threads) && (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
                       (proc_type_table[0][MAIN_CORE_PROC] > 0) && (n_threads > proc_type_table[0][MAIN_CORE_PROC])) {
                n_streams = (n_threads >= proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC])
                                ? static_cast<int>(n_threads - proc_type_table[0][EFFICIENT_CORE_PROC] / 2)
                                : static_cast<int>(proc_type_table[0][MAIN_CORE_PROC] +
                                                   (n_threads - proc_type_table[0][MAIN_CORE_PROC]) / 2);
                n_streams = (input_infer_requests > 0) ? std::min(n_streams, input_infer_requests) : n_streams;
                n_threads_per_stream = -1;
            } else {
                n_streams = ((n_threads + model_prefer_threads - 1) / model_prefer_threads);
                n_streams = (input_infer_requests > 0) ? std::min(n_streams, input_infer_requests) : n_streams;
                n_threads_per_stream = std::min(static_cast<int>(n_threads / n_streams), proc_type_table[0][base_type]);
            }
        }

        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;

        for (int n_type = MAIN_CORE_PROC; (n_type <= HYPER_THREADING_PROC) && (n_streams > 0); n_type++) {
            if (proc_type_table[0][n_type] > 0) {
                if (proc_type_table.size() == 1) {
                    update_streams_per_node(n_type, proc_type_table[0]);
                } else {
                    for (size_t n_node = 1; (n_node < proc_type_table.size()) && (n_streams > 0); n_node++) {
                        update_streams_per_node(n_type, proc_type_table[n_node]);
                    }
                }
            }
        }

        if (n_streams > 0) {
            for (int n_type = MAIN_CORE_PROC; n_type <= HYPER_THREADING_PROC; n_type++) {
                int proc_sum = 0;
                for (size_t n_socket = 0; n_socket < proc_socket_table.size(); n_socket++) {
                    if (proc_socket_table[n_socket][n_type] >= stream_info[THREADS_PER_STREAM]) {
                        stream_info[PROC_TYPE] = n_type;
                        stream_info[NUMBER_OF_STREAMS] =
                            static_cast<int>(proc_socket_table[n_socket][n_type] / stream_info[THREADS_PER_STREAM]);
                        stream_info[STREAM_NUMA_NODE_ID] = -1;
                        stream_info[STREAM_SOCKET_ID] = n_socket;
                        streams_info_table.push_back(stream_info);
                        n_streams -= stream_info[NUMBER_OF_STREAMS];
                        proc_socket_table[n_socket][n_type] -=
                            stream_info[THREADS_PER_STREAM] * stream_info[NUMBER_OF_STREAMS];
                        if (n_streams <= 0) {
                            break;
                        }
                    }
                    proc_sum += proc_socket_table[n_socket][n_type];
                }
                if (n_streams <= 0) {
                    break;
                }
                if (proc_sum >= stream_info[THREADS_PER_STREAM]) {
                    stream_info[PROC_TYPE] = n_type;
                    stream_info[NUMBER_OF_STREAMS] = static_cast<int>(proc_sum / stream_info[THREADS_PER_STREAM]);
                    stream_info[STREAM_NUMA_NODE_ID] = -1;
                    stream_info[STREAM_SOCKET_ID] = -1;
                    streams_info_table.push_back(stream_info);
                    n_streams -= stream_info[NUMBER_OF_STREAMS];
                    if (n_streams <= 0) {
                        break;
                    }
                }
            }
        }

        return streams_info_table;
    }
}

int get_model_prefer_threads(const int num_streams,
                             const std::vector<std::vector<int>> proc_type_table,
                             const std::shared_ptr<ngraph::Function>& ngraphFunc,
                             Config& config) {
    const int sockets = get_default_latency_streams(config.latencyThreadingMode);
    auto model_prefer = 0;
    if (-1 == config.modelPreferThreads) {
        const auto isa = dnnl::get_effective_cpu_isa();
        float isaSpecificThreshold = 1.0f;
        switch (isa) {
        case dnnl::cpu_isa::sse41:
            isaSpecificThreshold = 0.5f;
            break;
        case dnnl::cpu_isa::avx2:
        case dnnl::cpu_isa::avx512_core:
            isaSpecificThreshold = 1.0f;
            break;
        case dnnl::cpu_isa::avx512_core_vnni:
        case dnnl::cpu_isa::avx2_vnni:
            isaSpecificThreshold = 2.0f;
            break;
        case dnnl::cpu_isa::avx512_core_amx:
            isaSpecificThreshold = 4.0f;
            break;
        default:
            isaSpecificThreshold = 1.0f;
        }
        // the more "capable" the CPU in general, the more streams we may want to keep to keep it utilized
        const float memThresholdAssumeLimitedForISA = ov::MemBandwidthPressure::LIMITED / isaSpecificThreshold;
        const float L2_cache_size = dnnl::utils::get_cache_size(2 /*level*/, true /*per core */);
        ov::MemBandwidthPressure networkToleranceForLowCache =
            ov::MemBandwidthPressureTolerance(ngraphFunc, L2_cache_size, memThresholdAssumeLimitedForISA);
        config.modelPreferThreads = ov::threading::IStreamsExecutor::Config::StreamMode::DEFAULT;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) ||
                (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                config.modelPreferThreads = 1;
            }  // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            config.modelPreferThreads = 1;
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            config.modelPreferThreads = 2;
        }
        if (config.modelPreferThreads == 1 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0 && sockets == 1) {
            config.modelPreferThreads = 2;
        }
    }

    // latency
    if (num_streams <= sockets && num_streams > 0) {
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][MAIN_CORE_PROC] > 0) {
            bool fp_intesive = !ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(ngraphFunc);
            const int int8_threshold = 4;  // ~relative efficiency of the VNNI-intensive code for Big vs Little cores;
            const int fp32_threshold = 2;  // ~relative efficiency of the AVX2 fp32 code for Big vs Little cores;
            // by default the latency case uses (faster) Big cores only, depending on the compute ratio
            model_prefer = proc_type_table[0][MAIN_CORE_PROC] > (proc_type_table[0][EFFICIENT_CORE_PROC] /
                                                                 (fp_intesive ? fp32_threshold : int8_threshold))
                               ? proc_type_table[0][MAIN_CORE_PROC]
                               : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC];
        }
    } else {  // throughput
        model_prefer = config.modelPreferThreads;
    }

    return model_prefer;
}

std::vector<std::vector<int>> generate_stream_info(const int streams,
                                                   const std::shared_ptr<ngraph::Function>& ngraphFunc,
                                                   Config& config,
                                                   std::vector<std::vector<int>>& proc_type_table,
                                                   int preferred_nthreads_per_stream) {
    int model_prefer_threads = preferred_nthreads_per_stream;
    InferenceEngine::IStreamsExecutor::Config& executor_config = config.streamExecutorConfig;

    proc_type_table = apply_scheduling_core_type(config.schedulingCoreType, proc_type_table);
    proc_type_table = apply_hyper_threading(config.enableHyperThreading,
                                            config.changedHyperThreading,
                                            config.perfHintsConfig.ovPerfHint,
                                            proc_type_table);
    executor_config._cpu_reservation = get_cpu_pinning(config.enableCpuPinning,
                                                       config.changedCpuPinning,
                                                       streams,
                                                       executor_config._threadBindingType,
                                                       config.latencyThreadingMode,
                                                       proc_type_table);
    if (-1 == preferred_nthreads_per_stream) {
        model_prefer_threads = get_model_prefer_threads(streams, proc_type_table, ngraphFunc, config);
    }

    executor_config._streams_info_table = get_streams_info_table(executor_config._streams,
                                                                 executor_config._streams_changed,
                                                                 executor_config._threads,
                                                                 config.perfHintsConfig.ovPerfHintNumRequests,
                                                                 model_prefer_threads,
                                                                 config.perfHintsConfig.ovPerfHint,
                                                                 config.latencyThreadingMode,
                                                                 proc_type_table);
    return proc_type_table;
}

void get_num_streams(const int streams, const std::shared_ptr<ngraph::Function>& ngraphFunc, Config& config) {
    InferenceEngine::IStreamsExecutor::Config& executor_config = config.streamExecutorConfig;
    std::vector<std::vector<int>> proc_type_table = get_proc_type_table();

    generate_stream_info(streams, ngraphFunc, config, proc_type_table);

    executor_config = InferenceEngine::IStreamsExecutor::Config::reserve_cpu_threads(executor_config);
    executor_config._threadsPerStream = executor_config._streams_info_table[0][THREADS_PER_STREAM];
}

int get_default_latency_streams(Config::LatencyThreadingMode latency_threading_mode) {
    if (latency_threading_mode == Config::LatencyThreadingMode::PER_NUMA_NODE) {
        return get_num_sockets();
    } else if (latency_threading_mode == Config::LatencyThreadingMode::PER_SOCKET) {
        return get_num_numa_nodes();
    } else {
        return 1;
    }
}

}  // namespace intel_cpu
}  // namespace ov
