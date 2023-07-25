// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_test_utils.hpp"
#include "ie_ngraph_utils.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"
#include <cstdint>

namespace CPUTestUtils {
const char* CPUTestsBase::any_type = "any_type";

const char *CPUTestsBase::cpu_fmt2str(cpu_memory_format_t v) {
#define CASE(_fmt) case (cpu_memory_format_t::_fmt): return #_fmt;
    switch (v) {
        CASE(undef);
        CASE(ncw);
        CASE(nCw8c);
        CASE(nCw16c);
        CASE(nwc);
        CASE(nchw);
        CASE(nChw8c);
        CASE(nChw16c);
        CASE(nhwc);
        CASE(ncdhw);
        CASE(nCdhw8c);
        CASE(nCdhw16c);
        CASE(ndhwc);
        CASE(nc);
        CASE(x);
        CASE(ntc);
        CASE(ldgoi);
        CASE(ldoi);
    }
#undef CASE
    assert(!"unknown fmt");
    return "undef";
}

cpu_memory_format_t CPUTestsBase::cpu_str2fmt(const char *str) {
#define CASE(_fmt) do { \
    if (!strcmp(#_fmt, str) \
            || !strcmp("dnnl_" #_fmt, str)) \
        return _fmt; \
} while (0)
    CASE(undef);
    CASE(a);
    CASE(ab);
    CASE(abc);
    CASE(acb);
    CASE(aBc8b);
    CASE(aBc16b);
    CASE(abcd);
    CASE(acdb);
    CASE(aBcd8b);
    CASE(aBcd16b);
    CASE(abcde);
    CASE(acdeb);
    CASE(aBcde8b);
    CASE(aBcde16b);
    CASE(bac);
    CASE(abdc);
    CASE(abdec);
    CASE(ncw);
    CASE(nCw8c);
    CASE(nCw16c);
    CASE(nwc);
    CASE(nchw);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(nhwc);
    CASE(ncdhw);
    CASE(nCdhw8c);
    CASE(nCdhw16c);
    CASE(ndhwc);
    CASE(nc);
    CASE(x);
    CASE(tnc);
    CASE(ntc);
    CASE(ldnc);
    CASE(ldigo);
    CASE(ldgoi);
    CASE(ldio);
    CASE(ldoi);
    CASE(ldgo);
#undef CASE
    assert(!"unknown memory format");
    return undef;
}

std::string CPUTestsBase::fmts2str(const std::vector<cpu_memory_format_t> &fmts, const std::string &prefix) {
    std::string str;
    for (auto &fmt : fmts) {
        ((str += prefix) += cpu_fmt2str(fmt)) += ",";
    }
    if (!str.empty()) {
        str.pop_back();
    }
    return str;
}

std::string CPUTestsBase::impls2str(const std::vector<std::string> &priority) {
    std::string str;
    for (auto &impl : priority) {
        ((str += "cpu:") += impl) += ",";
    }
    if (!str.empty()) {
        str.pop_back();
    }
    return str;
}

void CPUTestsBase::CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, const std::set<std::string>& nodeType) const {
    if (!execNet || nodeType.empty()) return;

    ASSERT_TRUE(!selectedType.empty()) << "Node type is not defined.";
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    CheckPluginRelatedResultsImpl(function, nodeType);
}

void CPUTestsBase::CheckPluginRelatedResults(const ov::CompiledModel &execNet, const std::set<std::string>& nodeType) const {
    if (!execNet || nodeType.empty()) return;

    ASSERT_TRUE(!selectedType.empty()) << "Node type is not defined.";
    auto function = execNet.get_runtime_model();
    CheckPluginRelatedResultsImpl(function, nodeType);
}

void CPUTestsBase::CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, const std::string& nodeType) const {
    CheckPluginRelatedResults(execNet, std::set<std::string>{nodeType});
}

void CPUTestsBase::CheckPluginRelatedResults(const ov::CompiledModel &execNet, const std::string& nodeType) const {
    CheckPluginRelatedResults(execNet, std::set<std::string>{nodeType});
}

void CPUTestsBase::CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const {
    ASSERT_NE(nullptr, function);
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };
        auto getExecValueOutputsLayout = [] (const std::shared_ptr<ngraph::Node>& node) -> std::string {
            auto rtInfo = node->get_rt_info();
            auto it = rtInfo.find(ExecGraphInfoSerialization::OUTPUT_LAYOUTS);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };
        // skip policy
        auto should_be_skipped = [] (const ngraph::PartialShape &partialShape, cpu_memory_format_t fmt) {
            if (partialShape.is_dynamic()) {
                return false;
            }

            auto shape = partialShape.get_shape();
            bool skip_unsquized_1D = static_cast<size_t>(std::count(shape.begin(), shape.end(), 1)) == shape.size() - 1;
            bool permule_of_1 = (fmt == cpu_memory_format_t::nhwc || fmt == cpu_memory_format_t::ndhwc || fmt == cpu_memory_format_t::nwc) && shape[1] == 1;
            return skip_unsquized_1D || permule_of_1;
        };

        if (nodeType.count(getExecValue(ExecGraphInfoSerialization::LAYER_TYPE))) {
            ASSERT_LE(inFmts.size(), node->get_input_size());
            ASSERT_LE(outFmts.size(), node->get_output_size());
            for (size_t i = 0; i < inFmts.size(); i++) {
                const auto parentPort = node->input_values()[i];
                const auto port = node->inputs()[i];
                if ((parentPort.get_tensor_ptr() == port.get_tensor_ptr())) {
                    auto parentNode = parentPort.get_node_shared_ptr();
                    auto shape = parentNode->get_output_tensor(0).get_partial_shape();
                    auto actualInputMemoryFormat = getExecValueOutputsLayout(parentNode);

                    if (!should_be_skipped(shape, inFmts[i])) {
                        ASSERT_EQ(inFmts[i], cpu_str2fmt(actualInputMemoryFormat.c_str()));
                    }
                }
            }

            /* actual output formats are represented as a single string, for example 'fmt1' or 'fmt1, fmt2, fmt3'
             * convert it to the list of formats */
            auto getActualOutputMemoryFormats = [] (const std::string& fmtStr) -> std::vector<std::string> {
                std::vector<std::string> result;
                std::stringstream ss(fmtStr);
                std::string str;
                while (std::getline(ss, str, ',')) {
                    result.push_back(str);
                }
                return result;
            };

            auto actualOutputMemoryFormats = getActualOutputMemoryFormats(getExecValueOutputsLayout(node));

            bool isAllEqual = true;
            for (size_t i = 1; i < outFmts.size(); i++) {
                if (outFmts[i - 1] != outFmts[i]) {
                    isAllEqual = false;
                    break;
                }
            }
            size_t fmtsNum = outFmts.size();
            if (isAllEqual) {
                fmtsNum = fmtsNum == 0 ? 0 : 1;
            } else {
                ASSERT_EQ(fmtsNum, actualOutputMemoryFormats.size());
            }

            for (size_t i = 0; i < fmtsNum; i++) {
                const auto actualOutputMemoryFormat = getExecValue(ExecGraphInfoSerialization::OUTPUT_LAYOUTS);
                const auto shape = node->get_output_partial_shape(i);

                if (should_be_skipped(shape, outFmts[i]))
                    continue;
                ASSERT_EQ(outFmts[i], cpu_str2fmt(actualOutputMemoryFormats[i].c_str()));
            }

            auto primType = getExecValue(ExecGraphInfoSerialization::IMPL_TYPE);

            ASSERT_TRUE(primTypeCheck(primType)) << "primType is unexpected: " << primType << " Expected: " << selectedType;
        }
    }
}

bool CPUTestsBase::primTypeCheck(std::string primType) const {
    return selectedType.find(CPUTestsBase::any_type) != std::string::npos || std::regex_match(primType, std::regex(selectedType));
}

std::string CPUTestsBase::getTestCaseName(CPUSpecificParams params) {
    std::ostringstream result;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
    std::tie(inFmts, outFmts, priority, selectedType) = params;
    if (!inFmts.empty()) {
        auto str = fmts2str(inFmts, "");
        std::replace(str.begin(), str.end(), ',', '.');
        result << "_inFmts=" << str;
    }
    if (!outFmts.empty()) {
        auto str = fmts2str(outFmts, "");
        std::replace(str.begin(), str.end(), ',', '.');
        result << "_outFmts=" << str;
    }
    if (!selectedType.empty()) {
        result << "_primitive=" << selectedType;
    }
    return result.str();
}

CPUTestsBase::CPUInfo CPUTestsBase::getCPUInfo() const {
    return makeCPUInfo(inFmts, outFmts, priority);
}

#if defined(OV_CPU_WITH_ACL)
std::string CPUTestsBase::getPrimitiveType() const {
    return "acl";
}
#else
std::string CPUTestsBase::getPrimitiveType() const {
    std::string isaType;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        isaType = "jit_avx512";
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        isaType = "jit_avx2";
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        isaType = "jit_sse42";
    } else {
        isaType = "ref";
    }
    return isaType;
}

#endif

std::string CPUTestsBase::getISA(bool skip_amx) const {
    std::string isaType;
    if (!skip_amx && InferenceEngine::with_cpu_x86_avx512_core_amx()) {
        isaType = "avx512_amx";
    } else if (InferenceEngine::with_cpu_x86_avx512f()) {
        isaType = "avx512";
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        isaType = "avx2";
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        isaType = "sse42";
    } else {
        isaType = "";
    }
    return isaType;
}

static std::string setToString(const std::unordered_set<std::string> s) {
    if (s.empty())
        return {};

    std::string result;
    result.append("{");
    for (const auto& str : s) {
        result.append(str);
        result.append(",");
    }
    result.append("}");

    return result;
}

CPUTestsBase::CPUInfo
CPUTestsBase::makeCPUInfo(const std::vector<cpu_memory_format_t>& inFmts,
                          const std::vector<cpu_memory_format_t>& outFmts,
                          const std::vector<std::string>& priority) {
    CPUInfo cpuInfo;

    if (!inFmts.empty()) {
        cpuInfo.insert({ov::intel_cpu::InputMemoryFormats::get_type_info_static(),
                        ov::intel_cpu::InputMemoryFormats(fmts2str(inFmts, "cpu:"))});
    }
    if (!outFmts.empty()) {
        cpuInfo.insert({ov::intel_cpu::OutputMemoryFormats::get_type_info_static(),
                        ov::intel_cpu::OutputMemoryFormats(fmts2str(outFmts, "cpu:"))});
    }
    if (!priority.empty()) {
        cpuInfo.insert({"PrimitivesPriority", impls2str(priority)});
    }

    cpuInfo.insert({"enforceBF16evenForGraphTail", true});

    return cpuInfo;
}

std::shared_ptr<ngraph::Function>
CPUTestsBase::makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
                                 const std::shared_ptr<ngraph::Node> &lastNode, std::string name) {
   auto newLastNode = modifyGraph(ngPrc, params, lastNode);
   ngraph::ResultVector results;

   for (size_t i = 0; i < newLastNode->get_output_size(); i++)
        results.push_back(std::make_shared<ngraph::opset1::Result>(newLastNode->output(i)));

   return std::make_shared<ngraph::Function>(results, params, name);
}

std::shared_ptr<ngraph::Node>
CPUTestsBase::modifyGraph(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) {
    lastNode->get_rt_info() = getCPUInfo();
    return lastNode;
}

std::string CPUTestsBase::makeSelectedTypeStr(std::string implString, ngraph::element::Type_t elType) {
    implString.push_back('_');
    implString += InferenceEngine::details::convertPrecision(elType).name();
    return implString;
}

std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams> &paramsVector) {
    auto adjustBlockedFormatByIsa = [](std::vector<cpu_memory_format_t>& formats) {
        for (auto& format : formats) {
            if (format == nCw16c)
                format = nCw8c;
            if (format == nChw16c)
                format = nChw8c;
            if (format == nCdhw16c)
                format = nCdhw8c;
        }
    };

    std::vector<CPUSpecificParams> filteredParamsVector = paramsVector;

    if (!InferenceEngine::with_cpu_x86_avx512f()) {
        for (auto& param : filteredParamsVector) {
            adjustBlockedFormatByIsa(std::get<0>(param));
            adjustBlockedFormatByIsa(std::get<1>(param));
        }
    }

    return filteredParamsVector;
}

inline void CheckNumberOfNodesWithTypeImpl(std::shared_ptr<const ov::Model> function,
                                           const std::unordered_set<std::string>& nodeTypes,
                                           size_t expectedCount) {
    ASSERT_NE(nullptr, function);
    size_t actualNodeCount = 0;
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        if (nodeTypes.count(getExecValue(ExecGraphInfoSerialization::LAYER_TYPE))) {
            actualNodeCount++;
        }
    }

    ASSERT_EQ(expectedCount, actualNodeCount) << "Unexpected count of the node types '" << setToString(nodeTypes) << "' ";
}


void CheckNumberOfNodesWithTypes(InferenceEngine::ExecutableNetwork &execNet, const std::unordered_set<std::string>& nodeTypes, size_t expectedCount) {
    if (!execNet) return;

    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    std::shared_ptr<const ov::Model> function = execGraphInfo.getFunction();

    CheckNumberOfNodesWithTypeImpl(function, nodeTypes, expectedCount);
}

void CheckNumberOfNodesWithTypes(const ov::CompiledModel &compiledModel, const std::unordered_set<std::string>& nodeTypes, size_t expectedCount) {
    if (!compiledModel) return;

    std::shared_ptr<const ov::Model> function = compiledModel.get_runtime_model();

    CheckNumberOfNodesWithTypeImpl(function, nodeTypes, expectedCount);
}

void CheckNumberOfNodesWithType(const ov::CompiledModel &compiledModel, const std::string& nodeType, size_t expectedCount) {
    CheckNumberOfNodesWithTypes(compiledModel, {nodeType}, expectedCount);
}

void CheckNumberOfNodesWithType(InferenceEngine::ExecutableNetwork &execNet, const std::string& nodeType, size_t expectedCount) {
    CheckNumberOfNodesWithTypes(execNet, {nodeType}, expectedCount);
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("jit") != std::string::npos && !InferenceEngine::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !InferenceEngine::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx") != std::string::npos && !InferenceEngine::with_cpu_x86_avx())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !InferenceEngine::with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !InferenceEngine::with_cpu_x86_avx512f())
            continue;
        if (selectedTypeStr.find("amx") != std::string::npos && !InferenceEngine::with_cpu_x86_avx512_core_amx())
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}
} // namespace CPUTestUtils
