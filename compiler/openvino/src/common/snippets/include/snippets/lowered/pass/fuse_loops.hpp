// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface FuseLoops
 * @brief The pass fuses marking Loops. The transformations support the following fusions of loops:
 *
 *        - Upper Loop is fused into the Current Loop
 *             Loop_0 (Upper)                 |
 *                |               =>          |
 *             Loop_1 (Current)     Loop_0 + Loop_1 => new `Loop_1`
 *           * It's possible only if other consumers of Loop_0 are after Loop_1 in Linear IR.
 *             Because Upper Loop_0 will be explicitly moved before Current Loop_1 in linear IR,
 *             and we must save control dependency (to avoid cases when after fusion some consumers of Loop_0 are before this Loop)
 *
 *        - Lower Loop is fused into the Current Loop
 *              Loop_0 (Current)    Loop_0 + Loop_1 => new `Loop_0`
 *                |               =>           |
 *              Loop_1 (Lower)                 |
 *           * It's possible only if other parents of Loop_1 are before Loop_0 in Linear IR.
 *             Because Lower Loop_1 will be explicitly moved after Current Loop_0 in linear IR,
 *             and we must save control dependency (to avoid cases when after fusion some parents of Loop_1 are after this Loop)
 *
 *        The main conditions of possible fusion is the equal increments and the equal/broadcastable work amounts.
 * @ingroup snippets
 */
class FuseLoops : public Pass {
public:
    OPENVINO_RTTI("FuseLoops", "Pass")
    FuseLoops();
    bool run(LinearIR& linear_ir) override;

    // This method checks that all ports which connect lower and upper loops are incremented.
    // This helps to avoid fusing for the ports with incompleted data
    static bool loop_ports_are_compatible(const LinearIR::LoopManagerPtr& loop_manager,
                                          const size_t loop_lower_id,
                                          const size_t loop_upper_id);

private:
    static bool can_be_fused(const LinearIR::LoopManager::LoopInfoPtr& loop_current,
                             const LinearIR::LoopManager::LoopInfoPtr& loop_target);
    static bool fuse_upper_into_current(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager,
                                        const std::shared_ptr<ExpressionPort>& current_entry_point,
                                        size_t current_loop_id, size_t target_loop_id,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos);
    static bool fuse_lower_into_current(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager,
                                        const std::shared_ptr<ExpressionPort>& current_entry_point,
                                        size_t current_loop_id, size_t target_loop_id,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos);
    static void move(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id,
                     LinearIR::constExprIt loop_begin_pos, LinearIR::constExprIt loop_end_pos, LinearIR::constExprIt pos);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
