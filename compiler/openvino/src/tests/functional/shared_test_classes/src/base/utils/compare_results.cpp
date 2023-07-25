// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/ops.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace utils {

namespace {
void compare(const std::shared_ptr<ov::Node> &node,
             size_t port,
             const ov::runtime::Tensor &expected,
             const ov::runtime::Tensor &actual,
             double absThreshold,
             double relThreshold) {
    ov::test::utils::compare(expected, actual, absThreshold, relThreshold);
}

void compare(const std::shared_ptr<ov::op::v0::DetectionOutput> &node,
             size_t port,
             const ov::runtime::Tensor &expected,
             const ov::runtime::Tensor &actual,
             double absThreshold,
             double relThreshold) {
        ASSERT_EQ(expected.get_size(), actual.get_size());

        size_t expSize = 0;
        size_t actSize = 0;

        const float* expBuf = expected.data<const float>();
        const float* actBuf = actual.data<const float>();
        ASSERT_NE(expBuf, nullptr);
        ASSERT_NE(actBuf, nullptr);

        for (size_t i = 0; i < actual.get_size(); i+=7) {
            if (expBuf[i] == -1)
                break;
            expSize += 7;
        }
        for (size_t i = 0; i < actual.get_size(); i+=7) {
            if (actBuf[i] == -1)
                break;
            actSize += 7;
        }
        ASSERT_EQ(expSize, actSize);
        ov::test::utils::compare(expected, actual, 1e-2f, relThreshold);
}

template<typename T>
void compareResults(const std::shared_ptr<ov::Node> &node,
                    size_t port,
                    const ov::runtime::Tensor &expected,
                    const ov::runtime::Tensor &actual,
                    double absThreshold,
                    double relThreshold) {
    return compare(ngraph::as_type_ptr<T>(node), port, expected, actual, absThreshold, relThreshold);
}

} // namespace

CompareMap getCompareMap() {
    CompareMap compareMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), compareResults<NAMESPACE::NAME>},

#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset11_tbl.hpp"
#include "openvino/opsets/opset12_tbl.hpp"

#include "ov_ops/opset_private_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
    return compareMap;
}

} // namespace utils
} // namespace test
} // namespace ov
