// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "ie_system_conf.h"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <exec_graph_info.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include "ie_system_conf.h"

namespace CPUTestUtils {
    typedef enum {
        undef,
        a,
        ab,
        acb,
        aBc8b,
        aBc16b,
        abcd,
        acdb,
        aBcd8b,
        aBcd16b,
        abcde,
        acdeb,
        aBcde8b,
        aBcde16b,
        // RNN layouts
        abc,
        bac,
        abdc,
        abdec,

        x = a,
        nc = ab,
        ncw = abc,
        nchw = abcd,
        ncdhw = abcde,
        nwc = acb,
        nhwc = acdb,
        ndhwc = acdeb,
        nCw8c = aBc8b,
        nCw16c = aBc16b,
        nChw8c = aBcd8b,
        nChw16c = aBcd16b,
        nCdhw8c = aBcde8b,
        nCdhw16c = aBcde16b,
        // RNN layouts
        tnc = abc,
        /// 3D RNN data tensor in the format (batch, seq_length, input channels).
        ntc = bac,
        /// 4D RNN states tensor in the format (num_layers, num_directions,
        /// batch, state channels).
        ldnc = abcd,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        ///  input_channels, num_gates, output_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldigo = abcde,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels, input_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgoi = abdec,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_hidden_state, num_channels_in_recurrent_projection).
        ldio = abcd,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_recurrent_projection, num_channels_in_hidden_state).
        ldoi = abdc,
        /// 4D RNN bias tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgo = abcd,
    } cpu_memory_format_t;

    using CPUSpecificParams =  std::tuple<
        std::vector<cpu_memory_format_t>, // input memomry format
        std::vector<cpu_memory_format_t>, // output memory format
        std::vector<std::string>,         // priority
        std::string                       // selected primitive type
    >;

    enum class nodeType {
        convolution,
        convolutionBackpropData,
        groupConvolution,
        groupConvolutionBackpropData
    };

    inline std::string nodeType2PluginType(nodeType nt) {
        if (nt == nodeType::convolution) return "Convolution";
        if (nt == nodeType::convolutionBackpropData) return "Deconvolution";
        if (nt == nodeType::groupConvolution) return "Convolution";
        if (nt == nodeType::groupConvolutionBackpropData) return "Deconvolution";
        throw std::runtime_error("Undefined node type to convert to plug-in type node!");
    }

    inline std::string nodeType2str(nodeType nt) {
        if (nt == nodeType::convolution) return "Convolution";
        if (nt == nodeType::convolutionBackpropData) return "ConvolutionBackpropData";
        if (nt == nodeType::groupConvolution) return "GroupConvolution";
        if (nt == nodeType::groupConvolutionBackpropData) return "GroupConvolutionBackpropData";
        throw std::runtime_error("Undefined node type to convert to string!");
    }

class CPUTestsBase {
public:
    typedef std::map<std::string, ov::Any> CPUInfo;

public:
    static std::string getTestCaseName(CPUSpecificParams params);
    static const char *cpu_fmt2str(cpu_memory_format_t v);
    static cpu_memory_format_t cpu_str2fmt(const char *str);
    static std::string fmts2str(const std::vector<cpu_memory_format_t> &fmts, const std::string &prefix);
    static std::string impls2str(const std::vector<std::string> &priority);
    static CPUInfo makeCPUInfo(const std::vector<cpu_memory_format_t>& inFmts,
                               const std::vector<cpu_memory_format_t>& outFmts,
                               const std::vector<std::string>& priority);
   //TODO: change to setter method
    static std::string makeSelectedTypeStr(std::string implString, ngraph::element::Type_t elType);

    CPUInfo getCPUInfo() const;
    std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc,
                                                         ngraph::ParameterVector &params,
                                                         const std::shared_ptr<ngraph::Node> &lastNode,
                                                         std::string name);

    void CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, const std::set<std::string>& nodeType) const;
    void CheckPluginRelatedResults(const ov::CompiledModel &execNet, const std::set<std::string>& nodeType) const;
    void CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, const std::string& nodeType) const;
    void CheckPluginRelatedResults(const ov::CompiledModel &execNet, const std::string& nodeType) const;

    static const char* any_type;

protected:
    virtual void CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const;
    /**
     * @brief This function modifies the initial single layer test graph to add any necessary modifications that are specific to the cpu test scope.
     * @param ngPrc Graph precision.
     * @param params Graph parameters vector.
     * @param lastNode The last node of the initial graph.
     * @return The last node of the modified graph.
     */
    virtual std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                                      ngraph::ParameterVector &params,
                                                      const std::shared_ptr<ngraph::Node> &lastNode);

    virtual bool primTypeCheck(std::string primType) const;

protected:
    std::string getPrimitiveType() const;
    std::string getISA(bool skip_amx) const;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
};

// common parameters
const auto emptyCPUSpec = CPUSpecificParams{{}, {}, {}, {}};
const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuFP32PluginConfig =
        { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO } };
const std::map<std::string, std::string> cpuBF16PluginConfig =
        { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES } };


// utility functions
std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams>& paramsVector);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams);
void CheckNumberOfNodesWithType(const ov::CompiledModel &compiledModel, const std::string& nodeType, size_t expectedCount);
void CheckNumberOfNodesWithType(InferenceEngine::ExecutableNetwork &execNet, const std::string& nodeType, size_t expectedCount);
void CheckNumberOfNodesWithTypes(const ov::CompiledModel &compiledModel, const std::unordered_set<std::string>& nodeTypes, size_t expectedCount);
void CheckNumberOfNodesWithTypes(InferenceEngine::ExecutableNetwork &execNet, const std::unordered_set<std::string>& nodeTypes, size_t expectedCount);
} // namespace CPUTestUtils
