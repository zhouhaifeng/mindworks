// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/op/memory_access.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Load
 * @brief Generated during Lowering stage (convert_to_snippets_dialect) where explicit instructions should be emitted for data loading
 *        where number of elements to load is determined by "count" (Default value is "1" - to load one element)
 *        and memory offset for loading is determined by "offset" (Default value is "0" - to load starting from the first element)
 * @ingroup snippets
 */
class Load : public MemoryAccess {
public:
    OPENVINO_OP("Load", "SnippetsOpset", MemoryAccess);

    Load(const Output<Node>& x, const size_t count = 1lu, const size_t offset = 0lu);
    Load() = default;

    size_t get_offset() const { return get_input_offset(0); }
    size_t get_count() const { return get_input_count(0); }

    void set_offset(size_t offset) { set_input_offset(offset, 0); }
    void set_count(size_t count) { set_input_count(count, 0); }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

protected:
    void validate_memory_access_params() const;
};

/**
 * @interface LoadReshape
 * @brief It's just Load operation (and it's mapped on LoadEmitter during code generation) that allows to tweak
 *        shape propagation. We need it to keep correct shape propagation  when Transpose is decomposed to
 *        Load and Store. This is a temporary solution until tokenization of Reshape operation is supported.
 * @ingroup snippets
 */
class LoadReshape : public Load {
public:
    OPENVINO_OP("LoadReshape", "SnippetsOpset", Load);
    LoadReshape(const Output<Node>& x, size_t count = 1lu, const size_t offset = 0lu, std::vector<size_t> order = {});
    LoadReshape() = default;

    void set_offset(size_t offset) { set_output_offset(offset, 0); }
    void set_count(size_t count) { set_output_count(count, 0); }

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    std::vector<size_t> m_order;
};
} // namespace op
} // namespace snippets
} // namespace ov
