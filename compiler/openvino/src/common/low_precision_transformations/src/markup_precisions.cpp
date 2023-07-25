// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_precisions.hpp"

#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include "openvino/opsets/opset12.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "itt.hpp"

using namespace ngraph;

ngraph::pass::low_precision::MarkupPrecisions::MarkupPrecisions(
    const std::vector<PrecisionsRestriction>& restrictions,
    const std::vector<ngraph::element::Type>& defaultPrecisions) : defaultPrecisions(defaultPrecisions) {
    for (const auto& restriction : restrictions) {
        const auto it = restrictionsByOperation.find(restriction.operationType.name);
        OPENVINO_SUPPRESS_DEPRECATED_START
        if (it == restrictionsByOperation.end()) {
            Restriction r(restriction.specifyVersion);
            r.precisionsByVersion.emplace(
                restriction.operationType.version_id,
                Restriction::RestrictionByVersion(restriction.precisionsByPortsFunction, restriction.precisionsByPorts));
            restrictionsByOperation.emplace(restriction.operationType.name, r);
        } else {
            it->second.add(
                restriction.operationType.version_id,
                Restriction::RestrictionByVersion(restriction.precisionsByPortsFunction, restriction.precisionsByPorts));
        }
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
}

namespace {
void setRestriction(
    const std::shared_ptr<Node>& node,
    const pass::low_precision::PrecisionsRestriction::PrecisionsByPorts& precisionsByPorts) {
    if (precisionsByPorts.empty()) {
        // if available precisions for any port is empty then mark all input ports
        for (auto& input : node->inputs()) {
            auto& rt = input.get_rt_info();
            rt.emplace(
                    PrecisionsAttribute::get_type_info_static(),
                    PrecisionsAttribute(std::vector<element::Type>()));
        }
    } else {
        for (const auto& item : precisionsByPorts) {
            const auto attr = PrecisionsAttribute(item.second);
            for (const auto& port : item.first) {
                Input<Node> input = node->input(port);
                auto& rt = input.get_rt_info();
                auto precisionsAttribute = ngraph::pass::low_precision::getAttribute<PrecisionsAttribute>(input);
                if ((!precisionsAttribute.empty()) && (precisionsAttribute.as<PrecisionsAttribute>().value().empty())) {
                    return;
                }
                rt[PrecisionsAttribute::get_type_info_static()] = attr;
            }
        }
    }
}
} // namespace

bool ngraph::pass::low_precision::MarkupPrecisions::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(MarkupPrecisions);
    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0) {
            continue;
        }

        if (transformation_callback(node)) {
            continue;
        }

        if (const auto multiSubGraph = ov::as_type_ptr<ngraph::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < multiSubGraph->get_internal_subgraphs_size(); i++)
                run_on_model(multiSubGraph->get_function(i));
            continue;
        }

        // TODO: don't need to set restrictions for not supported operations
        // if don't set restrictions for not supported operations then accuracy drop appears, issue #59197
        const bool supported = ov::is_type<opset1::Result>(node) || isSupported(node);
        if (!supported && restrictionsByOperation.find(node->get_type_info().name) != restrictionsByOperation.end())
            THROW_IE_LPT_EXCEPTION(*node) << "Restriction is set for unsupported operation";
        if (!supported || !LayerTransformation::canBeTransformedStatic(node, defaultPrecisions)) {
            setRestriction(node, pass::low_precision::PrecisionsRestriction::PrecisionsByPorts{{{0ul}, {}}});
            continue;
        }

        const bool precisionPreserved = isPrecisionPreserved(node);
        if (precisionPreserved) {
            auto& rt = node->get_rt_info();
            rt.emplace(
                PrecisionPreservedAttribute::get_type_info_static(),
                PrecisionPreservedAttribute(precisionPreserved));
        }

        const auto& typeInfo = node->get_type_info();
        auto it = restrictionsByOperation.find(typeInfo.name);
        if (it != restrictionsByOperation.end()) {
            const Restriction& r = it->second;
            if (r.versionIsRequired) {
                const auto it2 = r.precisionsByVersion.find(typeInfo.version_id);
                if (it2 == r.precisionsByVersion.end()) {
                    continue;
                }

                const auto& precisionsByPorts = it2->second;
                setRestriction(node, precisionsByPorts.get(node));
            } else {
                assert(r.precisionsByVersion.size() == 1ul);

                const auto& precisionsByPorts = r.precisionsByVersion.begin()->second;
                setRestriction(node, precisionsByPorts.get(node));
            }
        }
    }
    return true;
}

template <class Operation>
std::string name() {
    return Operation::get_type_info_static().name;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isPrecisionPreserved(const std::shared_ptr<Node>& node) {
    if (isDisabled(node)) {
        return false;
    }

    // TODO: think how to handle conditions <= not mandatory for PoC
    // TODO: operation set version is not affected <= not mandatory for PoC
    static std::unordered_set<std::string> precisionPreservedOps = {
        { name<opset1::Concat>() },
        { name<opset1::DepthToSpace>() },
        { name<opset1::Interpolate>() },
        { name<opset1::MaxPool>() },
        { name<opset1::ReduceMax>() },
        { name<opset1::ReduceMin>() },
        { name<opset1::Relu>() },
        // TODO: there are conditions
        { name<opset1::Pad>() },
        { name<ov::opset12::Pad>() },
        { name<opset1::Reshape>() },
        { name<opset1::Squeeze>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() }
    };

    const bool precisionPreserved = precisionPreservedOps.find(node->get_type_name()) != precisionPreservedOps.end();
    if (precisionPreserved) {
        return precisionPreserved;
    }

    if (ov::is_type<opset1::Interpolate>(node)) {
        std::shared_ptr<opset1::Interpolate> interpolate1 = ov::as_type_ptr<opset1::Interpolate>(node);
        if (interpolate1) {
            const auto attrs = interpolate1->get_attrs();
            return attrs.mode == "nearest";
        }

        std::shared_ptr<opset4::Interpolate> interpolate4 = ov::as_type_ptr<opset4::Interpolate>(node);
        if (interpolate4) {
            const auto attrs = interpolate4->get_attrs();
            return attrs.mode == op::v4::Interpolate::InterpolateMode::NEAREST;
        }
    }

    return false;
}

bool ngraph::pass::low_precision::MarkupPrecisions::isSupported(const std::shared_ptr<Node>& node) {
    static std::unordered_set<std::string> supportedOps = {
        { name<opset1::Add>() },
        { name<opset1::AvgPool>() },
        { name<opset1::Clamp>() },
        { name<opset1::Concat>() },
        // ?
        { name<opset1::Convert>() },
        { name<opset1::Convolution>() },
        { name<opset1::ConvolutionBackpropData>() },
        { name<opset1::DepthToSpace>() },
        { name<opset1::FakeQuantize>() },
        { name<opset1::Interpolate>() },
        { name<opset4::Interpolate>() },
        { name<opset1::GroupConvolution>() },
        { name<opset1::MatMul>() },
        { name<opset1::MaxPool>() },
        { name<opset1::Multiply>() },
        { name<ngraph::op::MVN>() },
        { name<opset6::MVN>() },
        { name<opset1::NormalizeL2>() },
        { name<opset1::Pad>() },
        { name<ov::opset12::Pad>() },
        { name<opset1::PRelu>() },
        { name<opset1::ReduceMax>() },
        { name<opset1::ReduceMean>() },
        { name<opset1::ReduceMin>() },
        { name<opset1::ReduceSum>() },
        { name<opset1::Relu>() },
        // TODO: there are conditions
        { name<opset1::Reshape>() },
        { name<opset1::Squeeze>() },
        { name<opset1::ShuffleChannels>() },
        { name<opset1::Split>() },
        { name<opset1::StridedSlice>() },
        // ?
        { name<opset1::Subtract>() },
        { name<opset1::Transpose>() },
        { name<opset1::Unsqueeze>() },
        { name<opset1::VariadicSplit>() },
        { name<opset5::LSTMSequence>() },
        { name<opset6::GRUSequence>() },
    };

    return supportedOps.find(node->get_type_name()) != supportedOps.end();
}
