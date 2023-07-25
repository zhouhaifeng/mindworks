// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <common_test_utils/ngraph_test_utils.hpp>
#include "snippets/op/subgraph.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

using BlockedShapeVector = ov::snippets::op::Subgraph::BlockedShapeVector;

class DummyEmitter : public ov::snippets::Emitter {
public:
    // Here I pass Add to Emitter, but could be any other op, since it's ignored anyway.
    DummyEmitter(const std::vector<ov::Node::type_info_t>& custom_opset = {}) : ov::snippets::Emitter(std::make_shared<ov::op::v1::Add>()) {}
    void emit_code(const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&,
                   const std::vector<size_t>&) const override {}
    void emit_data() const override {}
};

class DummyTargetMachine : public ov::snippets::TargetMachine {
public:
    DummyTargetMachine(const std::vector<ov::Node::type_info_t>& custom_opset = {});
    bool is_supported() const override { return true; }
    ov::snippets::code get_snippet() const override { return nullptr; }
    size_t get_lanes() const override { return 10; }
};

class DummyGenerator : public ov::snippets::Generator {
public:
    DummyGenerator() : ov::snippets::Generator(std::make_shared<DummyTargetMachine>()) {}
    DummyGenerator(const std::shared_ptr<ov::snippets::TargetMachine>& t) : ov::snippets::Generator(t) {}

protected:
    opRegType get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const override { return vec2vec; };
};

class LoweringTests : public TransformationTestsF {
public:
    LoweringTests();

    void SetUp() override;
    void TearDown() override;

protected:
    static std::shared_ptr<ov::snippets::op::Subgraph> getSubgraph(const std::shared_ptr<Model>& f);
    static std::shared_ptr<ov::snippets::op::Subgraph> getLoweredSubgraph(const std::shared_ptr<Model>& f,
                                                                              const ov::PartialShape& master_shape,
                                                                              ov::pass::Manager pre_dialect = {},
                                                                              ov::pass::Manager post_dialect = {},
                                                                              ov::pass::Manager post_precision = {},
                                                                              ov::snippets::lowered::pass::PassPipeline lowered_pipeline = {},
                                                                              const std::shared_ptr<ov::snippets::Generator> generator = nullptr);
    static std::shared_ptr<ov::snippets::op::Subgraph> getTokenizedSubgraph(const std::shared_ptr<Model>& f);
    ov::PartialShape master_shape{};
};

}  // namespace snippets
}  // namespace test
}  // namespace ov