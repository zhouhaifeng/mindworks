// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/assign_registers.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

// This header is needed to avoid MSVC warning "C2039: 'inserter': is not a member of 'std'"
#include <iterator>

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool AssignRegisters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AssignRegisters")
    using Reg = size_t;
    using tensor = PortConnectorPtr;
    const auto& expressions = linear_ir.get_ops();

    std::vector<std::pair<Generator::opRegType, ExpressionPtr>> typed_ops;
    NodeVector ops;
    Reg num_parameters = 0;
    Reg num_results = 0;
    Reg num_expressions = 0;
    for (auto& expr : expressions) {
        auto op = expr->get_node();
        auto reg_type = m_reg_type_mapper(op);
        typed_ops.emplace_back(reg_type, expr);
        num_parameters += is_type<ov::op::v0::Parameter>(op);
        num_results += is_type<ov::op::v0::Result>(op);
        ops.push_back(op);
        num_expressions++;
    }
    size_t counter_vec = 0;
    size_t counter_gpr = 0;
    std::map<tensor, Reg> regs_vec, regs_gpr;
    // Define a set of immune tensors that will be ignored by auto reg allocation => their reg allocation is done manually
    std::map<tensor, Reg> manually_assigned_gprs, manually_assigned_vecs;
    const auto IS_MANUALLY_ALLOCATED_REG = SIZE_MAX;
    auto accumulator_reg = 0lu;
    for (const auto& expr : expressions) {
        auto op = expr->get_node();
        if (const auto io_expr = std::dynamic_pointer_cast<IOExpression>(expr)) {
            if (io_expr->get_type() == IOExpression::io_type::INPUT)
                manually_assigned_gprs[expr->get_output_port_connector(0)] = io_expr->get_index();
            else if (io_expr->get_type() == IOExpression::io_type::OUTPUT)
                manually_assigned_gprs[expr->get_input_port_connector(0)] = num_parameters + io_expr->get_index();
            else
                OPENVINO_THROW("Unsupported io_type detected");
        } else if (const auto& buffer = ov::as_type_ptr<op::Buffer>(op)) {
            const auto buffer_id = buffer->get_id();
            // All buffers have one common data pointer
            if (buffer->is_intermediate_memory()) {
                manually_assigned_gprs[expr->get_input_port_connector(0)] =
                        static_cast<Reg>(num_results + num_parameters + buffer_id);
            }
            manually_assigned_gprs[expr->get_output_port_connector(0)] =
                    static_cast<Reg>(num_results + num_parameters + buffer_id);
        } else if (ov::is_type<op::HorizonMax>(op) || ov::is_type<op::HorizonSum>(op)) {
            // Only in SoftmaxDecomposition ReduceMax and ReduceSum use HorizonMax/HorizonSum and VectorBuffer.
            // We should manually set the one vector register for VectorBuffer and Max/Sum output to simulate a accumulator
            // TODO [96351]: We should rewrite accumulator pattern using another way
            const auto& input_tensor = expr->get_input_port_connector(0);
            const auto& input_expr = input_tensor->get_source().get_expr();
            const auto& input_expr_input_tensors = input_expr->get_input_port_connectors();
            for (const auto& tensor : input_expr_input_tensors) {
                if (ov::is_type<op::VectorBuffer>(tensor->get_source().get_expr()->get_node())) {
                    manually_assigned_vecs[tensor] = static_cast<Reg>(accumulator_reg);
                }
            }
            const auto& output_tensor = expr->get_output_port_connector(0);
            manually_assigned_vecs[input_tensor] = static_cast<Reg>(accumulator_reg);
            manually_assigned_vecs[output_tensor] = static_cast<Reg>(accumulator_reg);
            for (const auto& child_expr_input : output_tensor->get_consumers()) {
                if (ov::is_type<op::BroadcastMove>(child_expr_input.get_expr()->get_node())) {
                    manually_assigned_vecs[child_expr_input.get_expr()->get_output_port_connector(0)] =
                            static_cast<Reg>(accumulator_reg);
                }
            }

            // TODO: Fix via common pipeline using LoopEnd:
            //       All operations `outside loop` after Horizon ops should have the same register to avoid using it in the next Loop
            const auto current_loops_ids = expr->get_loop_ids();
            auto next_expr = output_tensor->get_consumers().begin()->get_expr();
            while (next_expr->get_loop_ids() == current_loops_ids) {
                manually_assigned_vecs[next_expr->get_output_port_connector(0)] =
                        static_cast<Reg>(accumulator_reg);
                next_expr = next_expr->get_output_port_connector(0)->get_consumers().begin()->get_expr();
            }

            accumulator_reg++;
        }
    }
    // Note: have to specify default capture "=" due to MSVC bug (it doesn't capture const expressions implicitly)
    // Otherwise WIN build fails with "IS_MANUALLY_ALLOCATED_REG cannot be implicitly captured because no default capture mode has been specified"
    // the same problem with all the other lambdas in this file
    auto enumerate_out_tensors = [=] (const ExpressionPtr& expr,
                                      decltype(regs_vec)& reg_map,
                                      const std::map<tensor, Reg>& manually_assigned_regs,
                                      size_t& counter) {
        for (const auto& out_tensor : expr->get_output_port_connectors()) {
            // Note that some ops might have identical input&output tensors (Result and Tile* for ex.)
            // so we have to check that the tensor has not been enumerated already
            if (reg_map.count(out_tensor) == 0) {
                reg_map[out_tensor] = manually_assigned_regs.count(out_tensor) == 0 ? counter++ : IS_MANUALLY_ALLOCATED_REG;
            }
        }
    };
    for (const auto& t_op : typed_ops) {
        switch (t_op.first) {
            case Generator::opRegType::vec2vec:
            case Generator::opRegType::gpr2vec:
                enumerate_out_tensors(t_op.second, regs_vec, manually_assigned_vecs, counter_vec);
                break;
            case Generator::opRegType::gpr2gpr:
            case Generator::opRegType::vec2gpr:
                enumerate_out_tensors(t_op.second, regs_gpr, manually_assigned_gprs, counter_gpr);
                break;
        }
    }
    // todo: make one for gpr and one for vector
    std::vector<std::set<Reg>> used_gpr(num_expressions, std::set<Reg>()); // used = used as an input
    std::vector<std::set<Reg>> defined_gpr(num_expressions, std::set<Reg>()); // defined = used as output
    std::vector<std::set<Reg>> used_vec(num_expressions, std::set<Reg>());
    std::vector<std::set<Reg>> defined_vec(num_expressions, std::set<Reg>());

    auto tensor2reg = [=] (const std::vector<tensor>& tensors, const std::map<tensor, Reg>& reg_map) {
        std::set<Reg> result;
        for (const auto& t : tensors) {
            if (reg_map.count(t) == 0)
                OPENVINO_THROW("Assign registers: attempt to access not enumerated tensor");
            Reg reg_id = reg_map.at(t);
            if (reg_id != IS_MANUALLY_ALLOCATED_REG)
                result.insert(reg_id);
        }
        return result;
    };
    for (size_t i = 0; i < typed_ops.size(); i++) {
        const auto& t_op = typed_ops[i];
        std::vector<tensor> used_tensors, defined_tensors;
        for (const auto& in : t_op.second->get_input_port_connectors())
            used_tensors.push_back(in);
        for (const auto& out : t_op.second->get_output_port_connectors())
            defined_tensors.push_back(out);
        switch (t_op.first) {
            case Generator::opRegType::vec2vec:
                used_vec[i] = tensor2reg(used_tensors, regs_vec);
                defined_vec[i] = tensor2reg(defined_tensors, regs_vec);
                break;
            case Generator::opRegType::gpr2gpr:
                used_gpr[i] = tensor2reg(used_tensors, regs_gpr);
                defined_gpr[i] = tensor2reg(defined_tensors, regs_gpr);
                break;
            case Generator::opRegType::gpr2vec:
                used_gpr[i] = tensor2reg(used_tensors, regs_gpr);
                defined_vec[i] = tensor2reg(defined_tensors, regs_vec);
                break;
            case Generator::opRegType::vec2gpr:
                used_vec[i] = tensor2reg(used_tensors, regs_vec);
                defined_gpr[i] = tensor2reg(defined_tensors, regs_gpr);
                break;
        }
    }

    // define life intervals
    // liveOut[i] - regs that are live on exit from i-th (topologically ordered) operation
    // liveIn[i] - regs that are live on entering the i-th (topologically ordered) operation
    std::vector<std::set<Reg>> life_in_vec(std::move(used_vec));
    std::vector<std::set<Reg>> life_out_vec(typed_ops.size(), std::set<Reg>());
    std::vector<std::set<Reg>> life_in_gpr(std::move(used_gpr));
    std::vector<std::set<Reg>> life_out_gpr(typed_ops.size(), std::set<Reg>());

    // todo: this part if O(N*N), so it's slow for large subgraphs. Can we simplify it? At least add an early stopping criteria
    for (size_t i = 0; i < typed_ops.size(); i++) {
        for (size_t n = 0; n < typed_ops.size(); n++) {
            // Regs that are live on entering the operation = regs used by the op + (all other regs alive - regs defined by the op)
            // copy regs from lifeOut to lifeIn while ignoring regs in def
            std::set_difference(life_out_gpr[n].begin(), life_out_gpr[n].end(),
                                defined_gpr[n].begin(), defined_gpr[n].end(),
                                std::inserter(life_in_gpr[n], life_in_gpr[n].begin()));
            std::set_difference(life_out_vec[n].begin(), life_out_vec[n].end(),
                                defined_vec[n].begin(), defined_vec[n].end(),
                                std::inserter(life_in_vec[n], life_in_vec[n].begin()));
        }
        for (size_t n = 0; n < typed_ops.size(); n++) {
            const auto& expr = typed_ops[n].second;
            if (is_type<op::LoopEnd>(expr->get_node()) || is_type<ov::op::v0::Result>(expr->get_node()))
                continue;
            for (const auto& out : expr->get_output_port_connectors()) {
                for (const auto& child_expr_input : out->get_consumers()) {
                    const auto& child_expr = child_expr_input.get_expr();
                    auto child_it = linear_ir.begin();
                    std::advance(child_it, n);
                    size_t k = n;
                    while (child_it != linear_ir.end() && *child_it != child_expr) {
                        child_it++;
                        k++;
                    }
                    if (k == typed_ops.size())
                        OPENVINO_THROW("assign registers can't find target op in the body");
                    switch (typed_ops[k].first) {
                        case Generator::opRegType::vec2vec:
                        case Generator::opRegType::vec2gpr:
                            life_out_vec[n].insert(life_in_vec[k].begin(), life_in_vec[k].end());
                            break;
                        case Generator::opRegType::gpr2gpr:
                        case Generator::opRegType::gpr2vec:
                            life_out_gpr[n].insert(life_in_gpr[k].begin(), life_in_gpr[k].end());
                            break;
                    }
                }
            }
        }
    }
    struct by_starting {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    struct by_ending {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        }
    };
    // A variable live interval - is a range (start, stop) of op indexes, such that
    // the variable is alive within this range (defined but not used by the last user)
    std::map<std::pair<int, int>, Reg, by_starting> live_intervals_vec, live_intervals_gpr;

    std::reverse(life_in_vec.begin(), life_in_vec.end());
    std::reverse(life_in_gpr.begin(), life_in_gpr.end());
    auto find_last_use = [](decltype(life_in_gpr) life_in, int i) -> int {
        int ln = static_cast<int>(life_in.size()) - 1;
        for (auto& x : life_in) {
            if (x.find(i) != x.end()) {
                return ln;
            }
            ln--;
        }
        return i;
    };
    for (int i = 0; i < static_cast<int>(typed_ops.size()); i++) {
        for (const auto& def : defined_vec[i])
            live_intervals_vec[std::make_pair(i, find_last_use(life_in_vec, static_cast<int>(def)))] = def;
        for (const auto& def : defined_gpr[i])
            live_intervals_gpr[std::make_pair(i, find_last_use(life_in_gpr, static_cast<int>(def)))] = def;
    }

    auto linescan_assign_registers = [](const decltype(live_intervals_vec)& live_intervals,
                                        const std::set<Reg>& reg_pool) {
        // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
        // todo: do we need multimap? <=> can an op have two inputs from the same op?
        std::map<std::pair<int, int>, Reg, by_ending> active;
        // uniquely defined register => reused reg (reduced subset enabled by reg by reusage)
        std::map<Reg, Reg> register_map;
        std::stack<Reg> bank;
        // regs are stored in ascending order in reg_pool, so walk in reverse to assign them the same way
        for (auto rit = reg_pool.crbegin(); rit != reg_pool.crend(); rit++)
            bank.push(*rit);

        std::pair<int, int> interval, active_interval;
        Reg unique_reg, active_unique_reg;
        for (const auto& interval_reg : live_intervals) {
            std::tie(interval, unique_reg) = interval_reg;
            // check expired
            while (!active.empty()) {
                std::tie(active_interval, active_unique_reg) = *active.begin();
                // if end of active interval has not passed yet => stop removing actives since they are sorted by end
                if (active_interval.second >= interval.first) {
                    break;
                }
                active.erase(active_interval);
                bank.push(register_map[active_unique_reg]);
            }
            // allocate
            if (active.size() == reg_pool.size()) {
                // todo: if it is LoopBegin or LoopEnd that requires gpr, and we don't have any in the pool,
                //  then assign SIZE_MAX-1 as a flag to spill a reg inside emitter
                OPENVINO_THROW("can't allocate registers for a snippet ");
            } else {
                register_map[unique_reg] = bank.top();
                bank.pop();
                active.insert(interval_reg);
            }
        }
        return register_map;
    };
    // todo: vec_/gpr_pool are hardware-specific and should be provided by a backend, e.g. overloaded generator
    std::set<Reg> vec_pool;
    for (Reg i = 0; i < reg_count; i++)
        vec_pool.insert(i);
    std::set<Reg> gpr_pool(vec_pool);
    for (const auto& t_reg : manually_assigned_vecs)
        vec_pool.erase(t_reg.second);
    for (const auto& t_reg : manually_assigned_gprs)
        gpr_pool.erase(t_reg.second);
    auto unique2reused_map_vec = linescan_assign_registers(live_intervals_vec, vec_pool);
    auto unique2reused_map_gpr = linescan_assign_registers(live_intervals_gpr, gpr_pool);

    std::map<tensor, Reg> assigned_regs(std::move(manually_assigned_gprs));
    assigned_regs.insert(manually_assigned_vecs.begin(), manually_assigned_vecs.end());
    auto register_assigned_regs = [=, &assigned_regs](const std::map<tensor, Reg>& unique_regs, const std::map<Reg, Reg>& unique2reused) {
        for (const auto& reg : unique_regs) {
            if (reg.second == IS_MANUALLY_ALLOCATED_REG)
                continue;
            if (unique2reused.count(reg.second) == 0)
                OPENVINO_THROW("Assign registers failed to allocate register for a tensor");
            assigned_regs[reg.first] = unique2reused.at(reg.second);
        }
    };
    register_assigned_regs(regs_vec, unique2reused_map_vec);
    register_assigned_regs(regs_gpr, unique2reused_map_gpr);

    for (auto& t_op : typed_ops) {
        RegInfo rinfo;
        const auto& expr = t_op.second;
        for (const auto& in : expr->get_input_port_connectors()) {
            rinfo.first.push_back(assigned_regs[in]);
        }
        for (const auto& out : expr->get_output_port_connectors()) {
            rinfo.second.push_back(assigned_regs[out]);
        }
        t_op.second->set_reg_info(rinfo);
    }
    return false;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

