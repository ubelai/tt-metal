// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_softsign_bf16_abs_denominator_replay.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::softsign
{
template <typename Config, int Iterations = 32>
inline void calculate_abs_denominator()
{
    static_assert(Iterations == 32, "selected compact traversal owns one whole tile");
    static_assert(Config::kBodySlots == 19);
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    ttpoly::softsign::abs_denominator_bh_body<ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::softsign
