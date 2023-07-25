// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/properties/properties.hpp"

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/any.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regmodule_properties(py::module m) {
    // Top submodule
    py::module m_properties = m.def_submodule("properties", "openvino.runtime.properties submodule");

    // Submodule properties - enums
    py::enum_<ov::Affinity>(m_properties, "Affinity", py::arithmetic())
        .value("NONE", ov::Affinity::NONE)
        .value("CORE", ov::Affinity::CORE)
        .value("NUMA", ov::Affinity::NUMA)
        .value("HYBRID_AWARE", ov::Affinity::HYBRID_AWARE);

    // Submodule properties - properties
    wrap_property_RW(m_properties, ov::enable_profiling, "enable_profiling");
    wrap_property_RW(m_properties, ov::cache_dir, "cache_dir");
    wrap_property_RW(m_properties, ov::auto_batch_timeout, "auto_batch_timeout");
    wrap_property_RW(m_properties, ov::num_streams, "num_streams");
    wrap_property_RW(m_properties, ov::inference_num_threads, "inference_num_threads");
    wrap_property_RW(m_properties, ov::compilation_num_threads, "compilation_num_threads");
    wrap_property_RW(m_properties, ov::affinity, "affinity");
    wrap_property_RW(m_properties, ov::force_tbb_terminate, "force_tbb_terminate");
    wrap_property_RW(m_properties, ov::enable_mmap, "enable_mmap");

    wrap_property_RO(m_properties, ov::supported_properties, "supported_properties");
    wrap_property_RO(m_properties, ov::available_devices, "available_devices");
    wrap_property_RO(m_properties, ov::model_name, "model_name");
    wrap_property_RO(m_properties, ov::optimal_number_of_infer_requests, "optimal_number_of_infer_requests");
    wrap_property_RO(m_properties, ov::range_for_streams, "range_for_streams");
    wrap_property_RO(m_properties, ov::optimal_batch_size, "optimal_batch_size");
    wrap_property_RO(m_properties, ov::max_batch_size, "max_batch_size");
    wrap_property_RO(m_properties, ov::range_for_async_infer_requests, "range_for_async_infer_requests");

    // Submodule hint
    py::module m_hint =
        m_properties.def_submodule("hint", "openvino.runtime.properties.hint submodule that simulates ov::hint");

    // Submodule hint - enums
    py::enum_<ov::hint::Priority>(m_hint, "Priority", py::arithmetic())
        .value("LOW", ov::hint::Priority::LOW)
        .value("MEDIUM", ov::hint::Priority::MEDIUM)
        .value("HIGH", ov::hint::Priority::HIGH)
        .value("DEFAULT", ov::hint::Priority::DEFAULT);

    OPENVINO_SUPPRESS_DEPRECATED_START
    py::enum_<ov::hint::PerformanceMode>(m_hint, "PerformanceMode", py::arithmetic())
        .value("UNDEFINED", ov::hint::PerformanceMode::UNDEFINED)
        .value("LATENCY", ov::hint::PerformanceMode::LATENCY)
        .value("THROUGHPUT", ov::hint::PerformanceMode::THROUGHPUT)
        .value("CUMULATIVE_THROUGHPUT", ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);
    OPENVINO_SUPPRESS_DEPRECATED_END

    py::enum_<ov::hint::SchedulingCoreType>(m_hint, "SchedulingCoreType", py::arithmetic())
        .value("ANY_CORE", ov::hint::SchedulingCoreType::ANY_CORE)
        .value("PCORE_ONLY", ov::hint::SchedulingCoreType::PCORE_ONLY)
        .value("ECORE_ONLY", ov::hint::SchedulingCoreType::ECORE_ONLY);

    py::enum_<ov::hint::ExecutionMode>(m_hint, "ExecutionMode", py::arithmetic())
        .value("PERFORMANCE", ov::hint::ExecutionMode::PERFORMANCE)
        .value("ACCURACY", ov::hint::ExecutionMode::ACCURACY);

    // Submodule hint - properties
    wrap_property_RW(m_hint, ov::hint::inference_precision, "inference_precision");
    wrap_property_RW(m_hint, ov::hint::model_priority, "model_priority");
    wrap_property_RW(m_hint, ov::hint::performance_mode, "performance_mode");
    wrap_property_RW(m_hint, ov::hint::enable_cpu_pinning, "enable_cpu_pinning");
    wrap_property_RW(m_hint, ov::hint::scheduling_core_type, "scheduling_core_type");
    wrap_property_RW(m_hint, ov::hint::enable_hyper_threading, "enable_hyper_threading");
    wrap_property_RW(m_hint, ov::hint::execution_mode, "execution_mode");
    wrap_property_RW(m_hint, ov::hint::num_requests, "num_requests");
    wrap_property_RW(m_hint, ov::hint::model, "model");
    wrap_property_RW(m_hint, ov::hint::allow_auto_batching, "allow_auto_batching");

    // Submodule intel_cpu
    py::module m_intel_cpu =
        m_properties.def_submodule("intel_cpu",
                                   "openvino.runtime.properties.intel_cpu submodule that simulates ov::intel_cpu");

    // Submodule intel_cpu property
    wrap_property_RW(m_intel_cpu, ov::intel_cpu::denormals_optimization, "denormals_optimization");
    wrap_property_RW(m_intel_cpu,
                     ov::intel_cpu::sparse_weights_decompression_rate,
                     "sparse_weights_decompression_rate");

    // Submodule intel_gpu
    py::module m_intel_gpu =
        m_properties.def_submodule("intel_gpu",
                                   "openvino.runtime.properties.intel_gpu submodule that simulates ov::intel_gpu");

    wrap_property_RO(m_intel_gpu, ov::intel_gpu::device_total_mem_size, "device_total_mem_size");
    wrap_property_RO(m_intel_gpu, ov::intel_gpu::uarch_version, "uarch_version");
    wrap_property_RO(m_intel_gpu, ov::intel_gpu::execution_units_count, "execution_units_count");
    wrap_property_RO(m_intel_gpu, ov::intel_gpu::memory_statistics, "memory_statistics");

    wrap_property_RW(m_intel_gpu, ov::intel_gpu::enable_loop_unrolling, "enable_loop_unrolling");

    // Submodule hint (intel_gpu)
    py::module m_intel_gpu_hint = m_intel_gpu.def_submodule(
        "hint",
        "openvino.runtime.properties.intel_gpu.hint submodule that simulates ov::intel_gpu::hint");

    // `ThrottleLevel` enum is conflicting with `priorities.hint.Priority` in bindings.
    // `ov::intel_gpu::hint::ThrottleLevel` workaround proxy class:
    class ThrottleLevelProxy {};

    py::class_<ThrottleLevelProxy, std::shared_ptr<ThrottleLevelProxy>> m_throttle_level(
        m_intel_gpu_hint,
        "ThrottleLevel",
        "openvino.runtime.properties.intel_gpu.hint.ThrottleLevel that simulates ov::intel_gpu::hint::ThrottleLevel");

    m_throttle_level.attr("LOW") = ov::intel_gpu::hint::ThrottleLevel::LOW;
    m_throttle_level.attr("MEDIUM") = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
    m_throttle_level.attr("HIGH") = ov::intel_gpu::hint::ThrottleLevel::HIGH;
    m_throttle_level.attr("DEFAULT") = ov::intel_gpu::hint::ThrottleLevel::DEFAULT;

    wrap_property_RW(m_intel_gpu_hint, ov::intel_gpu::hint::queue_throttle, "queue_throttle");
    wrap_property_RW(m_intel_gpu_hint, ov::intel_gpu::hint::queue_priority, "queue_priority");
    wrap_property_RW(m_intel_gpu_hint, ov::intel_gpu::hint::host_task_priority, "host_task_priority");
    wrap_property_RW(m_intel_gpu_hint, ov::intel_gpu::hint::available_device_mem, "available_device_mem");

    // Submodule device
    py::module m_device =
        m_properties.def_submodule("device", "openvino.runtime.properties.device submodule that simulates ov::device");

    // Submodule device - enums
    py::enum_<ov::device::Type>(m_device, "Type", py::arithmetic())
        .value("INTEGRATED", ov::device::Type::INTEGRATED)
        .value("DISCRETE", ov::device::Type::DISCRETE);

    py::class_<ov::device::Priorities, std::shared_ptr<ov::device::Priorities>> cls_priorities(m_device, "Priorities");

    // Special case: ov::device::priorities
    m_device.def("priorities", []() {
        return ov::device::priorities.name();
    });

    m_device.def("priorities", [](py::args& args) {
        std::string value{""};
        for (auto v : args) {
            if (py::isinstance<py::str>(v)) {
                value += py::cast<std::string>(v) + std::string{','};
            } else {
                throw py::type_error("Incorrect passed value: " + std::string(py::str(v)) +
                                     " , expected string values.");
            }
        }
        return ov::device::priorities(value);
    });

    // Submodule device - properties
    wrap_property_RW(m_device, ov::device::id, "id");

    wrap_property_RO(m_device, ov::device::full_name, "full_name");
    wrap_property_RO(m_device, ov::device::architecture, "architecture");
    wrap_property_RO(m_device, ov::device::type, "type");
    wrap_property_RO(m_device, ov::device::gops, "gops");
    wrap_property_RO(m_device, ov::device::thermal, "thermal");
    wrap_property_RO(m_device, ov::device::capabilities, "capabilities");
    wrap_property_RO(m_device, ov::device::uuid, "uuid");

    // Special case: ov::device::properties
    m_device.def("properties", []() {
        return ov::device::properties.name();
    });

    m_device.def("properties", [](py::args& args) {
        ov::AnyMap value = {};
        for (auto v : args) {
            if (!py::isinstance<py::dict>(v)) {
                throw py::type_error("Incorrect passed value: " + std::string(py::str(v)) +
                                     ", expected dictionary instead of " + typeid(v).name());
            }
            auto dict = py::cast<py::dict>(v);
            for (auto item : dict) {
                if (!py::isinstance<py::str>(item.first)) {
                    throw py::type_error("Incorrect passed key in value: " + std::string(py::str(item.first)) +
                                         ", expected string instead of " + typeid(item.first).name());
                }
                value[py::cast<std::string>(item.first)] =
                    Common::utils::py_object_to_any(py::cast<py::object>(item.second));
            }
        }
        return ov::device::properties(value);
    });

    // Modules made in pybind cannot easily register attributes, thus workaround is needed.
    // Let's simulate module with attributes by creating empty proxy class called FakeModuleName.
    class FakeCapability {};

    py::class_<FakeCapability, std::shared_ptr<FakeCapability>> m_capability(
        m_device,
        "Capability",
        "openvino.runtime.properties.device.Capability that simulates ov::device::capability");

    m_capability.attr("FP32") = ov::device::capability::FP32;
    m_capability.attr("BF16") = ov::device::capability::BF16;
    m_capability.attr("FP16") = ov::device::capability::FP16;
    m_capability.attr("INT8") = ov::device::capability::INT8;
    m_capability.attr("INT16") = ov::device::capability::INT16;
    m_capability.attr("BIN") = ov::device::capability::BIN;
    m_capability.attr("WINOGRAD") = ov::device::capability::WINOGRAD;
    m_capability.attr("EXPORT_IMPORT") = ov::device::capability::EXPORT_IMPORT;

    // Submodule memory_type (intel_gpu)
    class FakeMemoryType {};

    py::class_<FakeMemoryType, std::shared_ptr<FakeMemoryType>> m_memory_type(
        m_intel_gpu,
        "MemoryType",
        "openvino.runtime.properties.intel_gpu.MemoryType submodule that simulates ov::intel_gpu::memory_type");

    m_memory_type.attr("surface") = ov::intel_gpu::memory_type::surface;
    m_memory_type.attr("buffer") = ov::intel_gpu::memory_type::buffer;

    // Submodule capability (intel_gpu)
    class FakeCapabilityGPU {};

    py::class_<FakeCapabilityGPU, std::shared_ptr<FakeCapabilityGPU>> m_capability_gpu(
        m_intel_gpu,
        "CapabilityGPU",
        "openvino.runtime.properties.intel_gpu.CapabilityGPU submodule that simulates ov::intel_gpu::capability");

    m_capability_gpu.attr("HW_MATMUL") = ov::intel_gpu::capability::HW_MATMUL;

    // Submodule log
    py::module m_log =
        m_properties.def_submodule("log", "openvino.runtime.properties.log submodule that simulates ov::log");

    // Submodule log - enums
    py::enum_<ov::log::Level>(m_log, "Level", py::arithmetic())
        .value("NO", ov::log::Level::NO)
        .value("ERR", ov::log::Level::ERR)
        .value("WARNING", ov::log::Level::WARNING)
        .value("INFO", ov::log::Level::INFO)
        .value("DEBUG", ov::log::Level::DEBUG)
        .value("TRACE", ov::log::Level::TRACE);

    // Submodule log - properties
    wrap_property_RW(m_log, ov::log::level, "level");

    // Submodule streams
    py::module m_streams =
        m_properties.def_submodule("streams",
                                   "openvino.runtime.properties.streams submodule that simulates ov::streams");

    py::class_<ov::streams::Num, std::shared_ptr<ov::streams::Num>> cls_num(m_streams, "Num");

    cls_num.def(py::init<>());
    cls_num.def(py::init<const int32_t>());

    // Covers static constexpr Num AUTO{-1};
    cls_num.attr("AUTO") = ov::streams::AUTO;
    // Covers static constexpr Num NUMA{-2};
    cls_num.attr("NUMA") = ov::streams::NUMA;

    cls_num.def("to_integer", [](ov::streams::Num& self) {
        return self.num;
    });

    // Submodule streams - properties RW
    wrap_property_RW(m_streams, ov::streams::num, "num");
    // Extra scenarios for ov::streams::num
    m_streams.def("num", [](const int32_t value) {
        return ov::streams::num(ov::streams::Num(value));
    });

    // Submodule auto
    py::module m_intel_auto =
        m_properties.def_submodule("intel_auto",
                                   "openvino.runtime.properties.intel_auto submodule that simulates ov::intel_auto");

    wrap_property_RW(m_intel_auto, ov::intel_auto::device_bind_buffer, "device_bind_buffer");
    wrap_property_RW(m_intel_auto, ov::intel_auto::enable_startup_fallback, "enable_startup_fallback");
    wrap_property_RW(m_intel_auto, ov::intel_auto::enable_runtime_fallback, "enable_runtime_fallback");
}
