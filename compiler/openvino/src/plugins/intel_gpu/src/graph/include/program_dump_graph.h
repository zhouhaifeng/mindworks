// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/program.hpp"
#include "program_node.h"
#include <fstream>
#include <string>

namespace cldnn {
std::string get_dir_path(const ExecutionConfig& config);

void dump_graph_optimized(std::ofstream&, const program&);
void dump_graph_processing_order(std::ofstream&, const program&);
void dump_graph_init(std::ofstream&, const program&, std::function<bool(program_node const&)> const&);
void dump_graph_info(std::ofstream&, const program&, std::function<bool(program_node const&)> const&);
}  // namespace cldnn
