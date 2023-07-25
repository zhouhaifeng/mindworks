// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/mvn.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <cmath>
#include <vector>

#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "low_precision/network_helper.hpp"

#include "ngraph/opsets/opset6.hpp"
#include "itt.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

namespace mvn {

template<typename T>
std::shared_ptr<ngraph::op::Constant> createNewScalesConst(const ngraph::op::Constant& originalConst) {
    std::vector<T> source = originalConst.cast_vector<T>();

    std::vector<T> newData(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        newData[i] = source[i] < 0 ? T{-1} : T{1};
    }

    const ngraph::element::Type type = originalConst.get_output_element_type(0);
    return ngraph::op::Constant::create(type, originalConst.get_shape(), newData);
}

} // namespace mvn

MVNTransformation::MVNTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(MVNTransformation);
    auto matcher = std::make_shared<pattern::op::Or>(OutputVector{
        pattern::wrap_type<ngraph::op::MVN>({ pattern::wrap_type<ov::opset1::Multiply>() }),
        pattern::wrap_type<ngraph::opset6::MVN>({ pattern::wrap_type<ov::opset1::Multiply>(), pattern::wrap_type<ov::opset1::Constant>() })
    });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool MVNTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(operation, defaultPrecisions);
    if (dequantization.empty() || dequantization.subtract != nullptr) {
        return false;
    }

    std::shared_ptr<Node> mvn = ov::as_type_ptr<op::MVN>(operation);
    if (!mvn) {
        mvn = ov::as_type_ptr<opset6::MVN>(operation);
        if (!mvn) {
            return false;
        }
    }

    AxisSet reduction_axes;
    if (ov::is_type<op::MVN>(mvn)) {
        reduction_axes = ov::as_type_ptr<op::MVN>(mvn)->get_reduction_axes();
    } else {
        // MVN-6 allows negative values in reduction axes: [-r, r-1]
        // given static rank of input data of MVN node, we can recover the exact axis number
        auto axis_set = ov::as_type_ptr<ov::opset1::Constant>(mvn->get_input_node_shared_ptr(1))->cast_vector<int64_t>();

        Dimension::value_type ndims = 0;
        if (std::any_of(axis_set.begin(), axis_set.end(), [](int64_t v) { return v < 0; })) {
            const auto rank = mvn->get_input_partial_shape(0).rank();
            // we need ndims to deduce exact axis if there are negative values
            if (rank.is_dynamic()) {
                return false;
            }
            ndims = rank.get_length();
        }

        for (auto& axis : axis_set) {
            reduction_axes.insert(axis >= 0 ? axis : axis + ndims);
        }
    }

    // scale-only-dequantization maybe per-channel or per-tensor
    // and it can be pushed-through MVN node only if there is one consistent
    // scale applied within a single normalization slice.
    // per-tensor scale-only-dequantization can always satisfy that
    if (NetworkHelper::isScalarLike(dequantization.multiplyConstant)) {
        return true;
    }

    // per-channel scale-only-dequantization can be pushed through MVN only
    // if the channel dimension is not among the reduction_axes (so a single
    // scale is applied to the whole normalization slice)
    if (reduction_axes.count(dequantization.channelDimIndex) == 0)
        return true;

    return false;
}

bool MVNTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<Node> operation = m.get_match_root();
    if (!canBeTransformed(context, operation)) {
        return false;
    }

    std::shared_ptr<Node> mvn = ov::as_type_ptr<op::MVN>(operation);
    if (!mvn) {
        mvn = ov::as_type_ptr<opset6::MVN>(operation);
    }

    bool normalizeVariance;
    if (ov::is_type<op::MVN>(mvn)) {
        normalizeVariance = ov::as_type_ptr<op::MVN>(mvn)->get_normalize_variance();
    } else {
        normalizeVariance = ov::as_type_ptr<opset6::MVN>(mvn)->get_normalize_variance();
    }

    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(mvn, defaultPrecisions);
    const auto scalesConst = dequantization.multiplyConstant;
    const auto type = scalesConst->get_element_type();

    auto newScalesConst = scalesConst;
    if (normalizeVariance) {
        switch (type) {
            case ngraph::element::Type_t::f16: {
                newScalesConst = mvn::createNewScalesConst<ngraph::element_type_traits<ngraph::element::Type_t::f16>::value_type>(*scalesConst);
                break;
            }
            case ngraph::element::Type_t::f32: {
                newScalesConst = mvn::createNewScalesConst<ngraph::element_type_traits<ngraph::element::Type_t::f32>::value_type>(*scalesConst);
                break;
            }
            default: {
                THROW_TRANSFORMATION_EXCEPTION << "unexpected element type " << type;
            }
        }
    }

    std::shared_ptr<Node> newMVN;
    if (ov::is_type<op::MVN>(mvn)) {
        newMVN = mvn->clone_with_new_inputs({dequantization.data});
    } else {
        newMVN = mvn->clone_with_new_inputs({dequantization.data, mvn->input_value(1)});
    }
    NetworkHelper::setOutDataPrecisionForTypeRelaxed(newMVN, deqPrecision);
    NetworkHelper::copyInfo(mvn, newMVN);

    auto newMultiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        ov::opset1::Multiply(newMVN, newScalesConst),
        mvn->get_output_element_type(0));
    ov::copy_runtime_info({ mvn, newMultiply }, newMultiply);

    NetworkHelper::insertDequantizationAfter(mvn, newMultiply, newMVN);

    updateOutput(context, newMultiply, newMVN);
    return true;
}

bool MVNTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
