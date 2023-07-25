// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <random>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector make_random_normal(const NodeContext& context, Output<Node> sizes, element::Type target_type) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> distrib(0, 9999);

    const uint64_t global_seed = 0;

    const uint64_t seed_1 = distrib(gen);
    const uint64_t seed_2 = distrib(gen);

    auto min_val = context.mark_node(v0::Constant::create(target_type, Shape{1}, {0}));
    auto max_val = context.mark_node(v0::Constant::create(target_type, Shape{1}, {1}));

    auto uniform_1 = context.mark_node(
        std::make_shared<v8::RandomUniform>(sizes, min_val, max_val, target_type, global_seed, seed_1));
    auto uniform_2 = context.mark_node(
        std::make_shared<v8::RandomUniform>(sizes, min_val, max_val, target_type, global_seed, seed_2));

    // Compute Box–Muller transform
    // random_normal = scale * ng.sqrt(-2.0 * ng.log(uniform_1)) * ng.cos(2.0 * np.pi * uniform_2) + mean
    auto pi = context.mark_node(v0::Constant::create(target_type, Shape{1}, {3.141592653589793}));
    auto minus_two = context.mark_node(v0::Constant::create(target_type, Shape{1}, {-2.0}));
    auto two = context.mark_node(v0::Constant::create(target_type, Shape{1}, {2.0}));

    auto log = context.mark_node(std::make_shared<v0::Log>(uniform_1));
    auto multiply_minus_two_log = context.mark_node(std::make_shared<v1::Multiply>(log, minus_two));
    auto sqrt = context.mark_node(std::make_shared<v0::Sqrt>(multiply_minus_two_log));

    auto multiply_two_pi = context.mark_node(std::make_shared<v1::Multiply>(uniform_2, pi));
    auto multiply_two_pi_uniform_2 = context.mark_node(std::make_shared<v1::Multiply>(multiply_two_pi, uniform_2));
    auto cos = context.mark_node(std::make_shared<v0::Cos>(multiply_two_pi_uniform_2));

    auto scale_const = context.mark_node(v0::Constant::create(target_type, Shape{1}, {1}));
    auto mean_const = context.mark_node(v0::Constant::create(target_type, Shape{1}, {0}));
    auto sqrt_x_cos = context.mark_node(std::make_shared<v1::Multiply>(sqrt, cos));
    auto product = context.mark_node(std::make_shared<v1::Multiply>(scale_const, sqrt_x_cos));
    auto sum = context.mark_node(std::make_shared<v1::Add>(product, mean_const));

    return {sum};
}
};  // namespace

OutputVector translate_rand(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto sizes = context.get_input(0);
    if (context.get_input_type(0).is<type::List>()) {
        sizes = concat_list_construct(sizes);
    }
    sizes = context.mark_node(std::make_shared<v0::Convert>(sizes, element::i32));
    auto low = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    auto high = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));
    auto dtype = element::f32;
    size_t out_id = 1;
    if (context.get_input_size() == 3) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(1),
                                      "aten::randn conversion with generator does not supported");
        out_id = 2;
    }
    // aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
    // aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
    if (context.get_input_size() == 2 || context.get_input_size() == 3) {
        auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
        context.mutate_input(out_id, res);
        return {res};
    }
    // aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    // aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None,
    // Device? device=None, bool? pin_memory=None) -> Tensor
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    size_t dtype_id = 1;
    if (context.get_input_size() == 6) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(1),
                                      "aten::rand conversion with generator does not supported");
        dtype_id = 2;
    }
    if (!context.input_is_none(dtype_id)) {
        if (std::dynamic_pointer_cast<v0::Constant>(
                context.get_input_from_visible_context(dtype_id).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(dtype_id));
            low = context.mark_node(std::make_shared<v0::Convert>(low, dtype));
            high = context.mark_node(std::make_shared<v0::Convert>(high, dtype));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(dtype_id)).get_node_shared_ptr(),
                                    "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
    if (!dtype_applied) {
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, convert_like_out));
    }
    return {res};
};

OutputVector translate_rand_like(const NodeContext& context) {
    // aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None, MemoryFormat? memory_format=None) -> Tensor aten::rand_like.out(Tensor self, *,
    // MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 6);
    auto inp_tensor = context.get_input(0);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(inp_tensor, element::i32));
    auto low = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    auto high = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));
    auto dtype = element::f32;
    if (context.get_input_size() == 3) {
        auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
        context.mutate_input(2, res);
        return {res};
    }
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    if (!context.input_is_none(1)) {
        if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(1).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(1));
            low = context.mark_node(std::make_shared<v0::Convert>(low, dtype));
            high = context.mark_node(std::make_shared<v0::Convert>(high, dtype));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(1)).get_node_shared_ptr(), "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
    if (!dtype_applied) {
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, convert_like_out));
    }
    return {res};
};

OutputVector translate_randn(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto sizes = context.get_input(0);
    if (context.get_input_type(0).is<type::List>()) {
        sizes = concat_list_construct(sizes);
    }
    sizes = context.mark_node(std::make_shared<v0::Convert>(sizes, element::i32));
    auto low = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    auto high = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));
    auto dtype = element::f32;
    size_t out_id = 1;
    if (context.get_input_size() == 3) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(1),
                                      "aten::randn conversion with generator does not supported");
        out_id = 2;
    }
    // aten::randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
    // aten::randn.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
    if (context.get_input_size() == 2 || context.get_input_size() == 3) {
        auto res = make_random_normal(context, sizes, dtype);
        context.mutate_input(out_id, res[0]);
        return res;
    }
    size_t dtype_id = 1;
    if (context.get_input_size() == 6) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(1),
                                      "aten::randn conversion with generator does not supported");
        dtype_id = 2;
    }
    // aten::randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    // aten::randn.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None,
    // Device? device=None, bool? pin_memory=None) -> Tensor
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    if (!context.input_is_none(dtype_id)) {
        if (std::dynamic_pointer_cast<v0::Constant>(
                context.get_input_from_visible_context(dtype_id).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(dtype_id));
            low = context.mark_node(std::make_shared<v0::Convert>(low, dtype));
            high = context.mark_node(std::make_shared<v0::Convert>(low, dtype));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(dtype_id)).get_node_shared_ptr(),
                                    "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto res = make_random_normal(context, sizes, dtype);
    if (!dtype_applied) {
        res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], convert_like_out));
    }
    return res;
};

OutputVector translate_randn_like(const NodeContext& context) {
    // aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    // aten::rand_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 6);
    auto inp_tensor = context.get_input(0);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(inp_tensor, element::i32));
    auto low = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    auto high = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));
    auto dtype = element::f32;
    if (context.get_input_size() == 3) {
        auto res = make_random_normal(context, sizes, dtype);
        context.mutate_input(2, res[0]);
        return res;
    }
    // aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    if (!context.input_is_none(1)) {
        if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(1).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(1));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(1)).get_node_shared_ptr(), "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto res = make_random_normal(context, sizes, dtype);
    if (!dtype_applied) {
        res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], convert_like_out));
    }
    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
