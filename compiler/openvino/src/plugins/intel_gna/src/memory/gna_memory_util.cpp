// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_memory_util.hpp"

#include <cstdint>

#include "log/debug.hpp"

namespace ov {
namespace intel_gna {
namespace memory {

int32_t MemoryOffset(void* ptr_target, void* ptr_base) {
    auto target = reinterpret_cast<uintptr_t>(ptr_target);
    auto base = reinterpret_cast<uintptr_t>(ptr_base);
    if (target == 0) {  // handle NULL pointers separately
        return (-1);
    } else if (target < base) {
        THROW_GNA_EXCEPTION << "Target address value " << target << " is less than base address " << base;
    } else {
        uint64_t diff = target - base;
        if (diff > 0x7fffffff) {
            THROW_GNA_EXCEPTION << "Target address value " << target << " too far from base address " << base;
        }
        return static_cast<int32_t>(diff);
    }
}

}  // namespace memory
}  // namespace intel_gna
}  // namespace ov
