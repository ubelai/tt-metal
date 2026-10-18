// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_softsign_bf16_abs_denominator_replay.h"
#include "sfpu/ckernel_sfpu_softsign_bf16_abs_denominator.h"
namespace ckernel::sfpu::bf16::softsign {
template <typename Config>
inline void init_abs_denominator() {
    sfpu_reciprocal_init<false>();
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kBoundBits);
    ttpoly::softsign::abs_denominator_loadmacro_init();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16::softsign
