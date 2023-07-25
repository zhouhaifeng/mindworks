// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined _WIN32 && !defined NOMINMAX
#define NOMINMAX
#endif

#include <functional>
#include <xbyak/xbyak.h>

namespace ngraph
{
    namespace runtime
    {
        namespace jit
        {
            class Generator : public Xbyak::CodeGenerator
            {
                static constexpr size_t xmm_len = 16;

#ifdef _WIN32
                static constexpr size_t xmm_to_preserve_start = 6;
                static constexpr size_t xmm_to_preserve = 10;
#else
                static constexpr size_t xmm_to_preserve_start = 0;
                static constexpr size_t xmm_to_preserve = 0;
#endif

                static const size_t num_abi_save_gpr_regs;
                const size_t size_of_abi_save_regs;

                const Xbyak::Reg64 reg_EVEX_max_8b_offt;
                static constexpr int EVEX_max_8b_offt = 0x200;

            public:
                static const Xbyak::Reg64 param;

                typedef enum
                {
                    isa_any,
                    sse42,
                    avx,
                    avx2,
                    avx512_common,
                    avx512_core,
                    avx512_core_vnni,
                    avx512_mic,
                    avx512_mic_4ops,
                    avx512_core_bf16,
                    avx512_vpopcnt,
                    fp16
                } cpu_isa_t;

                static bool mayiuse(const cpu_isa_t cpu_isa);
                static bool is_x64();

                Generator(void* code_ptr = nullptr, size_t code_size = 16 * 1024);
                void preamble();
                void postamble();

                void foreach (const Xbyak::Reg64& idx,
                              size_t step,
                              const Xbyak::Reg64& end,
                              std::function<void(const Xbyak::Reg64&)> && fn);

                template <typename T>
                void copy(const Xbyak::Reg64& dst,
                          const Xbyak::Reg64& src,
                          const Xbyak::Reg64& size);
            };
        }
    }
}
