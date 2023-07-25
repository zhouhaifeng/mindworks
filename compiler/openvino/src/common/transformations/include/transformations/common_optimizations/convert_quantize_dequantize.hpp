// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertQuantizeDequantize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertQuantizeDequantize transformation replaces following graph:
 * FakeQuantize->Convert->Convert->Subtract->Multiply with a single FakeQuantize.
 * Restrictions:
 * - quantized data type must be i8 or u8
 * - 'levels' attribute to FakeQuantize must be equal to 256
 * - (output_low, output_high) must be (-128, 127) or (0, 256) (depends on sign of quantized data type)
 * - 'zero_point' and 'scale' must be broadcastable to FakeQuantize's output
 */

class ov::pass::ConvertQuantizeDequantize : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertQuantizeDequantize", "0");
    ConvertQuantizeDequantize();
};
