// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <xml_parse_utils.h>

#include <ir_deserializer.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <openvino/op/util/framework_node.hpp>
#include <pugixml.hpp>

#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset.hpp"

using namespace ngraph;
using namespace InferenceEngine;

namespace {
void parse_pre_process(pugi::xml_node& root,
                       std::shared_ptr<ngraph::runtime::AlignedBuffer> weights,
                       std::shared_ptr<Function> f) {
    /* Preprocessing block can have two preprocessing types:
     *
     * <pre-process mean-precision="FP32" reference-layer-name="data">
     *     <channel id="0">
     *         <mean value="1.1"/>
     *     </channel>
     * </pre-process>
     *
     * OR
     *
     * <pre-process mean-precision="FP32" reference-layer-name="data">
     *     <channel id="0">
     *         <mean offset="0" size="1936"/>
     *     </channel>
     * </pre-process>
     */

    auto ppNode = root.child("pre-process");
    if (ppNode.empty()) {
        return;
    }
    // find out to what input this belongs to
    std::string inputName;
    std::shared_ptr<Node> input_node;

    inputName = pugixml::utils::GetStrAttr(ppNode, "reference-layer-name", "");
    inputName = trim(inputName);

    if (inputName.empty()) {
        // fallback (old format), look for the picture in the inputs
        for (const auto& parameter : f->get_parameters()) {
            if (parameter->get_partial_shape().rank().is_static() &&
                parameter->get_partial_shape().rank().get_length() == 4) {
                input_node = parameter;
                break;
            }
        }

        IE_ASSERT(!f->get_parameters().empty());
        if (!input_node) {
            input_node = f->get_parameters()[0];
        }

        inputName = input_node->get_friendly_name();
    } else {
        for (const auto& parameter : f->get_parameters()) {
            if (parameter->get_friendly_name() == inputName) {
                input_node = parameter;
                break;
            }
        }
    }

    if (!input_node)
        IE_THROW() << "pre-process name ref '" << inputName << "' refers to un-existing input";

    const auto& input_shape = input_node->output(0).get_partial_shape();
    if (input_shape.is_dynamic()) {
        IE_THROW() << "can not apply pre-process for '" << inputName << "' input";
    }

    Shape mean_scalar_shape;  // [C, 1 ... 1]
    Shape mean_shape;         // [1, H, W] - for 4D case

    const auto inputDims = input_shape.to_shape();

    if (inputDims.size() < 2) {
        IE_THROW() << "network did not define input dimensions properly";
    } else if (inputDims.size() == 2) {  // NC
        mean_scalar_shape = {inputDims[1]};
        mean_shape = {1};
    } else if (inputDims.size() == 3) {  // CHW - legacy representation for 3D input shape
        mean_scalar_shape = {inputDims[0], 1, 1};
        mean_shape = {1, inputDims[1], inputDims[2]};
    } else if (inputDims.size() == 4) {  // NCHW
        mean_scalar_shape = {inputDims[1], 1, 1};
        mean_shape = {1, inputDims[2], inputDims[3]};
    } else if (inputDims.size() == 5) {  // NCDHW
        mean_scalar_shape = {inputDims[1], 1, 1, 1};
        mean_shape = {1, inputDims[2], inputDims[3], inputDims[4]};
    }
    const size_t channels = mean_scalar_shape[0];

    uint64_t next_channel_id{0};
    std::set<std::pair<uint64_t, float>> mean_scalar_values;
    std::set<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> mean_values;

    auto input_type = input_node->get_output_element_type(0);
    FOREACH_CHILD (chan, ppNode, "channel") {
        auto chanNo = pugixml::utils::GetUInt64Attr(chan, "id", next_channel_id++);

        auto meanNode = chan.child("mean");
        if (!meanNode.empty()) {
            if (!meanNode.attribute("value") && (!meanNode.attribute("size"))) {
                IE_THROW() << "mean should have at least one of the following attribute: value, size";
            }
            if (meanNode.attribute("value")) {
                mean_scalar_values.insert({chanNo, pugixml::utils::GetFloatAttr(meanNode, "value")});
            }
            if (meanNode.attribute("size") && meanNode.attribute("offset")) {
                auto const_size = pugixml::utils::GetUInt64Attr(meanNode, "size");
                auto const_offset = pugixml::utils::GetUInt64Attr(meanNode, "offset");
                if (shape_size(mean_shape) * input_type.size() != const_size) {
                    IE_THROW() << "mean blob size mismatch expected input, got: " << const_size << " expecting "
                               << mean_shape << " x " << input_type.size();
                }
                if (const_offset + const_size > weights->size()) {
                    IE_THROW() << "mean value offset and size are out of weights size range";
                }
                mean_values.insert({chanNo, {const_size, const_offset}});
            }
        }
    }

    if (!mean_values.empty() && !mean_scalar_values.empty()) {
        IE_THROW() << "mean values have different types";
    }

    if (!mean_scalar_values.empty()) {
        if (mean_scalar_values.size() != channels) {
            IE_THROW() << "Number of mean values (" << mean_scalar_values.size()
                       << ") is not equal to number of channels (" << channels << ")";
        }
        std::vector<float> values(channels);
        for (const auto& item : mean_scalar_values) {
            if (item.first >= channels) {
                IE_THROW() << "Mean values channel index " << item.first << " is out of range (" << channels << ")";
            }
            values[item.first] = item.second;
        }
        auto mean_values_constant = ngraph::op::Constant::create(input_type, mean_scalar_shape, values);

        const auto& consumers = input_node->output(0).get_target_inputs();
        auto add = std::make_shared<ngraph::opset1::Subtract>(input_node, mean_values_constant);
        for (const auto& consumer : consumers) {
            consumer.replace_source_output(add);
        }
    }

    if (!mean_values.empty()) {
        if (mean_values.size() != channels) {
            IE_THROW() << "Number of mean values (" << mean_values.size() << ") is not equal to number of channels ("
                       << channels << ")";
        }
        NodeVector per_channel_values(channels);
        for (const auto& item : mean_values) {
            if (item.first >= channels) {
                IE_THROW() << "Mean values channel index " << item.first << " is out of range (" << channels << ")";
            }
            const size_t offset = item.second.second;
            const char* data = weights->get_ptr<char>() + offset;
            per_channel_values[item.first] = ngraph::opset1::Constant::create(input_type, mean_shape, data);
        }
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto const_node = get_constant_from_source(std::make_shared<ngraph::opset1::Concat>(per_channel_values, 0));
        OPENVINO_SUPPRESS_DEPRECATED_END
        IE_ASSERT(const_node);
        const auto& consumers = input_node->output(0).get_target_inputs();
        auto add = std::make_shared<ngraph::opset1::Subtract>(input_node, const_node);
        for (const auto& consumer : consumers) {
            consumer.replace_source_output(add);
        }
    }
}
}  // namespace

namespace ov {
namespace frontend {
namespace ir {

class InputModel::InputModelIRImpl {
    std::shared_ptr<ngraph::runtime::AlignedBuffer> m_weights;
    std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> m_extensions;
    std::unordered_map<std::string, ov::OpSet> m_opsets;
    pugi::xml_node m_root;
    pugi::xml_document m_xml_doc;

public:
    InputModelIRImpl(std::istream& stream,
                     const std::shared_ptr<ngraph::runtime::AlignedBuffer>& weights,
                     const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions)
        : m_weights(weights),
          m_extensions(extensions) {
        pugi::xml_parse_result res = m_xml_doc.load(stream);
        if (res.status != pugi::status_ok) {
            IE_THROW() << res.description() << " at offset " << res.offset;
        }
        m_root = m_xml_doc.document_element();
        for (const auto& it : ov::get_available_opsets()) {
            m_opsets[it.first] = it.second();
        }
    }

    std::shared_ptr<Function> convert();
};

InputModel::InputModel(std::istream& stream,
                       const std::shared_ptr<ngraph::runtime::AlignedBuffer>& weights,
                       const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions) {
    _impl = std::make_shared<InputModelIRImpl>(stream, weights, extensions);
}

std::shared_ptr<Function> InputModel::convert() {
    return _impl->convert();
}

std::shared_ptr<Function> InputModel::InputModelIRImpl::convert() {
    std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>> variables;

    // Load default opsets
    size_t version = pugixml::utils::GetUIntAttr(m_root, "version", 0);
    ov::XmlDeserializer visitor(m_root, m_weights, m_opsets, m_extensions, variables, version);
    std::shared_ptr<ngraph::Function> function;
    visitor.on_attribute("net", function);
    function->get_rt_info()["version"] = int64_t(version);
    parse_pre_process(m_root, m_weights, function);

    return function;
}

}  // namespace ir
}  // namespace frontend
}  // namespace ov
