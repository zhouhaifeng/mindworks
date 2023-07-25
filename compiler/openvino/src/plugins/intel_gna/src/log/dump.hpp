// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ios>
#include <sstream>
#include <string>

#include "gna2-model-api.h"
#include "gna_device.hpp"

namespace ov {
namespace intel_gna {
namespace dump {

void WriteInputAndOutputTextGNAImpl(const Gna2Model& gnaModel,
                                    const std::string dumpFolderNameGNA,
                                    const std::string refFolderName);

void DumpGna2Model(const Gna2Model& gnaModel,
                   const std::string& dumpFolderNameGNA,
                   bool dumpData,
                   const GnaAllocations& allAllocations,
                   const std::string& modeOfOperation);

}  // namespace dump
}  // namespace intel_gna
}  // namespace ov
