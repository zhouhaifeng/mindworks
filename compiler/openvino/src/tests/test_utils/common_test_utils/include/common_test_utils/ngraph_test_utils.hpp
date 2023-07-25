// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "common_test_utils/test_common.hpp"
#include "transformations/init_node_info.hpp"

#define DYN ov::Dimension::dynamic()

using TransformationTests = ov::test::TestsCommon;

class TransformationTestsF : public ov::test::TestsCommon {
public:
    TransformationTestsF();

    void SetUp() override;

    void TearDown() override;

    // TODO: this is temporary solution to disable rt info checks that must be applied by default
    // first tests must be fixed then this method must be removed XXX-68696
    void disable_rt_info_check();

    void enable_soft_names_comparison();
    void disable_result_friendly_names_check();

    std::shared_ptr<ov::Model> function, function_ref;
    // Aliases to function and function_ref pointers to be more corresponding with ov namespace.
    std::shared_ptr<ov::Model>&model, &model_ref;
    ov::pass::Manager manager;
    FunctionsComparator comparator;

private:
    std::shared_ptr<ov::pass::UniqueNamesHolder> m_unh;
    bool m_disable_rt_info_check{false};
    bool m_soft_names_comparison{true};
    bool m_result_friendly_names_check{true};
};

void init_unique_names(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::pass::UniqueNamesHolder>& unh);

void check_unique_names(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::pass::UniqueNamesHolder>& unh);

template <typename T>
size_t count_ops_of_type(const std::shared_ptr<ov::Model>& f) {
    size_t count = 0;
    for (auto op : f->get_ops()) {
        if (ov::is_type<T>(op)) {
            count++;
        }
    }

    return count;
}

template <class T>
std::shared_ptr<ov::op::v0::Constant> create_constant(const std::vector<T>& data,
                                                      const ov::element::Type_t et = ov::element::i64,
                                                      bool scalar = false) {
    ov::Shape shape = scalar ? ov::Shape{} : ov::Shape{data.size()};
    return ov::op::v0::Constant::create(et, shape, data);
}

std::shared_ptr<ov::op::v0::Constant> create_zero_constant(const ov::element::Type_t& et, const ov::Shape& shape);
