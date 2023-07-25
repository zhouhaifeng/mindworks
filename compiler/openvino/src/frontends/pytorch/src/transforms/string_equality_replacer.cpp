// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_equality_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

StringEqualityReplacer::StringEqualityReplacer() {
    auto framework_node_lhs = pattern::wrap_type<PtFrameworkNode>();
    auto framework_node_rhs = pattern::wrap_type<PtFrameworkNode>();
    auto convert_lhs = pattern::wrap_type<v0::Convert>({framework_node_lhs});
    auto convert_like_lhs = pattern::wrap_type<v1::ConvertLike>({framework_node_lhs, framework_node_rhs});
    auto convert_rhs = pattern::wrap_type<v0::Convert>({framework_node_rhs});
    auto convert_like_rhs = pattern::wrap_type<v1::ConvertLike>({framework_node_rhs, framework_node_lhs});
    auto lhs_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{framework_node_lhs, convert_lhs, convert_like_lhs});
    auto rhs_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{framework_node_rhs, convert_rhs, convert_like_rhs});
    auto equal_op = pattern::wrap_type<v1::Equal>({lhs_pattern, rhs_pattern});
    auto not_equal_op = pattern::wrap_type<v1::NotEqual>({lhs_pattern, rhs_pattern});

    auto string_equality_pattern = std::make_shared<pattern::op::Or>(OutputVector{equal_op, not_equal_op});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();

        auto lhs_node =
            std::dynamic_pointer_cast<PtFrameworkNode>(pattern_map.at(framework_node_lhs).get_node_shared_ptr());
        if (!lhs_node) {
            return false;
        }
        auto lhs_attrs = lhs_node->get_attrs();
        if (lhs_attrs.find("string_value") == lhs_attrs.end()) {
            return false;
        }
        std::string lhs = lhs_attrs.at("string_value");

        auto rhs_node =
            std::dynamic_pointer_cast<PtFrameworkNode>(pattern_map.at(framework_node_rhs).get_node_shared_ptr());
        if (!rhs_node) {
            return false;
        }
        auto rhs_attrs = rhs_node->get_attrs();
        if (rhs_attrs.find("string_value") == rhs_attrs.end()) {
            return false;
        }
        std::string rhs = rhs_attrs.at("string_value");

        auto equal_node = pattern_map.at(equal_op).get_node_shared_ptr();
        if (auto equal = std::dynamic_pointer_cast<v1::Equal>(equal_node)) {
            auto const_result = v0::Constant::create(element::boolean, Shape{}, {lhs == rhs});
            copy_runtime_info_and_name(equal_node, {const_result});
            replace_node(equal_node, const_result);
            return true;
        };
        auto not_equal_node = pattern_map.at(not_equal_op).get_node_shared_ptr();
        if (auto equal = std::dynamic_pointer_cast<v1::NotEqual>(not_equal_node)) {
            auto const_result = v0::Constant::create(element::boolean, Shape{}, {lhs != rhs});
            copy_runtime_info_and_name(equal_node, {const_result});
            replace_node(equal_node, const_result);
            return true;
        };
        return false;
    };
    auto m = std::make_shared<pattern::Matcher>(string_equality_pattern,
                                                "ov::frontend::pytorch::pass::StringEqualityReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
