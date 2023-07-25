// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Bucketize : public Node {
public:
    Bucketize(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }

    void prepareParams() override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename T, typename T_BOUNDARIES, typename T_IND>
    void bucketize();

    const size_t INPUT_TENSOR_PORT = 0;
    const size_t INPUT_BINS_PORT = 1;
    const size_t OUTPUT_TENSOR_PORT = 0;

    size_t num_values = 0;
    size_t num_bin_values = 0;
    bool with_right = false;
    bool with_bins = false;

    InferenceEngine::Precision input_precision;
    InferenceEngine::Precision boundaries_precision;
    InferenceEngine::Precision output_precision;
    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
