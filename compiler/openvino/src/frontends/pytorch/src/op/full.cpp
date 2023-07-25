// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> base_translate_full(const NodeContext& context, const Output<Node>& sizes, const Output<Node>& value) {
    return context.mark_node(std::make_shared<v3::Broadcast>(value, sizes));
}

Output<Node> base_translate_full_with_convertlike(const NodeContext& context,
                                                  const Output<Node>& sizes,
                                                  const Output<Node>& value,
                                                  const Output<Node>& out) {
    auto filled_tensor = base_translate_full(context, sizes, value);
    return context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, out));
}

Output<Node> base_translate_full_with_convert(const NodeContext& context,
                                              const Output<Node>& sizes,
                                              Output<Node> value,
                                              size_t dtype_id) {
    if (!context.input_is_none(dtype_id)) {
        value = apply_dtype(context, dtype_id, value);
    }

    auto filled_tensor = base_translate_full(context, sizes, value);
    return filled_tensor;
}
}  // namespace

OutputVector translate_full(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto sizes = context.get_input(0);
    auto value = context.get_input(1);
    auto num_inputs = context.get_input_size();
    if (num_inputs < 6) {
        int out_id = num_inputs == 3 ? 2 : 3;
        if (!context.input_is_none(static_cast<size_t>(out_id))) {
            auto out = context.get_input(out_id);
            return {base_translate_full_with_convertlike(context, sizes, value, out)};
        }
        return {base_translate_full(context, sizes, value)};
    }
    size_t dtype_id = num_inputs == 6 ? 2 : 3;
    return {base_translate_full_with_convert(context, sizes, value, dtype_id)};
};

OutputVector translate_full_like(const NodeContext& context) {
    num_inputs_check(context, 2, 7);
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    if (context.get_input_size() == 7 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    auto out = context.input_is_none(3) ? input : context.get_input(3);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_fill_(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_new_full(const NodeContext& context) {
    num_inputs_check(context, 3, 7);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.get_input(2);
    if (context.get_input_size() == 7 && !context.input_is_none(3)) {
        return {base_translate_full_with_convert(context, sizes, value, 3)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_zeros(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto sizes = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto num_inputs = context.get_input_size();
    if (num_inputs < 5) {
        int out_id = num_inputs == 2 ? 1 : 2;
        if (!context.input_is_none(static_cast<size_t>(out_id))) {
            auto out = context.get_input(out_id);
            return {base_translate_full_with_convertlike(context, sizes, value, out)};
        }
        return {base_translate_full(context, sizes, value)};
    }
    size_t dtype_id = num_inputs == 5 ? 1 : 2;
    return {base_translate_full_with_convert(context, sizes, value, dtype_id)};
};

OutputVector translate_zeros_like(const NodeContext& context) {
    num_inputs_check(context, 1, 6);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    if (context.get_input_size() == 6 && !context.input_is_none(1)) {
        return {base_translate_full_with_convert(context, sizes, value, 1)};
    }
    auto out = context.input_is_none(2) ? input : context.get_input(2);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_new_zeros(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    if (context.get_input_size() == 6 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_ones(const NodeContext& context) {
    num_inputs_check(context, 1, 5);
    auto sizes = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto num_inputs = context.get_input_size();
    if (num_inputs < 5) {
        int out_id = num_inputs == 2 ? 1 : 2;
        if (!context.input_is_none(static_cast<size_t>(out_id))) {
            auto out = context.get_input(out_id);
            return {base_translate_full_with_convertlike(context, sizes, value, out)};
        }
        return {base_translate_full(context, sizes, value)};
    }
    size_t dtype_id = num_inputs == 5 ? 1 : 2;
    return {base_translate_full_with_convert(context, sizes, value, dtype_id)};
};

OutputVector translate_ones_like(const NodeContext& context) {
    num_inputs_check(context, 1, 6);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    if (context.get_input_size() == 6 && !context.input_is_none(1)) {
        return {base_translate_full_with_convert(context, sizes, value, 1)};
    }
    auto out = context.input_is_none(2) ? input : context.get_input(2);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_new_ones(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    if (context.get_input_size() == 6 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_empty(const NodeContext& context) {
    // aten::empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None, MemoryFormat? memory_format=None) -> Tensor layout, device and work with memory ignored on our
    // side, so just skip these parameters
    num_inputs_check(context, 1, 6);
    auto sizes = context.get_input(0);
    // In OV uninitialized data is not supported, so we create a tensor filled with zeros with a given shape and type.
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    int dtype_id = 1;
    Output<Node> empty;
    if (!context.input_is_none(dtype_id)) {
        empty = base_translate_full_with_convert(context, sizes, value, dtype_id);
    } else {
        empty = base_translate_full(context, sizes, value);
    }
    return {empty};
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
