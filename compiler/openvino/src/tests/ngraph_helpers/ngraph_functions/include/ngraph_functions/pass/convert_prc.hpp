// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace ngraph {
namespace pass {

template<ngraph::element::Type_t from, ngraph::element::Type_t to>
class ConvertConstantsPrecision : public MatcherPass {
public:
    ConvertConstantsPrecision() {
        auto constant =
            std::make_shared<ngraph::op::Constant>(element::f32, Shape{1}, std::vector<float>{0});

        ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
            auto constant = std::dynamic_pointer_cast<ngraph::op::Constant>(m.get_match_root());
            if (!constant) {
                return false;
            }

            if (constant->get_element_type() == ngraph::element::Type(from)) {
                auto data = constant->cast_vector<typename ngraph::helpers::nGraphTypesTrait<to>::value_type>();
                auto new_const = std::make_shared<ngraph::op::Constant>(to, constant->get_shape(), data);
                new_const->set_friendly_name(constant->get_friendly_name());
                ngraph::replace_node(constant, new_const);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(constant, "ConvertConstantsPrecision");
        register_matcher(m, callback);
    }
};

template<ngraph::element::Type_t from, ngraph::element::Type_t to>
class ConvertParametersPrecision : public MatcherPass {
public:
    ConvertParametersPrecision() {
        auto constant = std::make_shared<ngraph::op::Parameter>(to, Shape{1});

        ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
            auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(m.get_match_root());
            if (parameter && parameter->get_element_type() == ngraph::element::Type(from)) {
                parameter->set_element_type(to);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(constant, "ConvertParametersPrecision");
        register_matcher(m, callback);
    }
};

template<ngraph::element::Type_t from, ngraph::element::Type_t to>
class ConvertConvertLayerOutputPrecision : public MatcherPass {
public:
    ConvertConvertLayerOutputPrecision() {
        auto convert = ngraph::pattern::wrap_type<opset1::Convert>();
        ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
            auto convert = std::dynamic_pointer_cast<ngraph::op::Convert>(m.get_match_root());
            if (!convert) {
                return false;
            }

            if (convert->get_convert_element_type() == ngraph::element::Type(from)) {
                convert->set_convert_element_type(to);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(convert, "ConvertConvertLayerPrecision");
        register_matcher(m, callback);
    }
};

template<ngraph::element::Type_t from, ngraph::element::Type_t to>
class ConvertPrecision : public ngraph::pass::GraphRewrite {
public:
    ConvertPrecision() {
        add_matcher<ConvertConstantsPrecision<from, to>>();
        add_matcher<ConvertParametersPrecision<from, to>>();
        add_matcher<ConvertConvertLayerOutputPrecision<from, to>>();
    }
};
}  // namespace pass
}  // namespace ngraph
