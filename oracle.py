import numpy as np
import numba
import config


@numba.jit(nopython=True, cache=True, fastmath=True)
def _generate_targets_jit(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    max_hold,
    fee_per_side=0.001,
    slippage=0.0005,
    sl_atr_mult=1.5,
    tp_atr_mult=3.0,
    enable_trailing=False,
    trail_atr_mult=1.5,
    saturation_factor=2.5,
    mae_penalty=0.20,
):
    """
    Oracle 5.0: Entry-ATR bracket exits with optional trailing overlay.

    Exit model:
    - Initial SL and TP are frozen at entry using entry ATR.
    - Optional trailing stop can tighten risk after favorable movement.
    - Intrabar handling is conservative when both SL and TP are touched
      in the same bar.
    """
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    total_cost_pct = (fee_per_side + slippage) * 2.0

    sl_distances = atr_arr * sl_atr_mult
    tp_distances = atr_arr * tp_atr_mult
    trail_distances = atr_arr * trail_atr_mult

    for i in range(n - max_hold):
        entry_price = close_arr[i]
        sl_dist = sl_distances[i]
        tp_dist = tp_distances[i]
        trail_dist = trail_distances[i]

        if sl_dist <= 0.0 or tp_dist <= 0.0 or entry_price <= 0.0:
            continue

        risk_pct = sl_dist / entry_price
        if risk_pct <= 0.0:
            continue

        cost_r = total_cost_pct / risk_pct
        if cost_r > 1.0:
            continue

        # ---------------- LONG LOGIC ----------------
        long_stop = entry_price - sl_dist
        long_tp = entry_price + tp_dist
        peak_price = entry_price
        max_risk_consumed_long = 0.0
        long_pnl_pct = 0.0

        for k in range(1, max_hold):
            idx = i + k
            c_open = open_arr[idx]
            c_high = high_arr[idx]
            c_low = low_arr[idx]
            c_close = close_arr[idx]

            # Gap handling
            if c_open <= long_stop:
                exit_price = c_open
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                break
            if c_open >= long_tp:
                exit_price = c_open
                long_pnl_pct = (exit_price - entry_price) / entry_price
                break

            # Optional trailing update only after favorable movement
            if enable_trailing and c_high > peak_price:
                peak_price = c_high
                new_stop = peak_price - trail_dist
                if new_stop > long_stop:
                    long_stop = new_stop

            # Intrabar resolution: conservative tie-breaker
            hit_sl = c_low <= long_stop
            hit_tp = c_high >= long_tp

            if hit_sl and hit_tp:
                exit_price = long_stop
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                break
            elif hit_sl:
                exit_price = long_stop
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                break
            elif hit_tp:
                exit_price = long_tp
                long_pnl_pct = (exit_price - entry_price) / entry_price
                break

            current_risk_consumed = (entry_price - c_low) / sl_dist
            if current_risk_consumed < 0.0:
                current_risk_consumed = 0.0
            elif current_risk_consumed > 1.0:
                current_risk_consumed = 1.0
            if current_risk_consumed > max_risk_consumed_long:
                max_risk_consumed_long = current_risk_consumed

            if k == max_hold - 1:
                long_pnl_pct = (c_close - entry_price) / entry_price

        # ---------------- SHORT LOGIC ----------------
        short_stop = entry_price + sl_dist
        short_tp = entry_price - tp_dist
        trough_price = entry_price
        max_risk_consumed_short = 0.0
        short_pnl_pct = 0.0

        for k in range(1, max_hold):
            idx = i + k
            c_open = open_arr[idx]
            c_high = high_arr[idx]
            c_low = low_arr[idx]
            c_close = close_arr[idx]

            # Gap handling
            if c_open >= short_stop:
                exit_price = c_open
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                break
            if c_open <= short_tp:
                exit_price = c_open
                short_pnl_pct = (entry_price - exit_price) / entry_price
                break

            # Optional trailing update only after favorable movement
            if enable_trailing and c_low < trough_price:
                trough_price = c_low
                new_stop = trough_price + trail_dist
                if new_stop < short_stop:
                    short_stop = new_stop

            # Intrabar resolution: conservative tie-breaker
            hit_sl = c_high >= short_stop
            hit_tp = c_low <= short_tp

            if hit_sl and hit_tp:
                exit_price = short_stop
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                break
            elif hit_sl:
                exit_price = short_stop
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                break
            elif hit_tp:
                exit_price = short_tp
                short_pnl_pct = (entry_price - exit_price) / entry_price
                break

            current_risk_consumed = (c_high - entry_price) / sl_dist
            if current_risk_consumed < 0.0:
                current_risk_consumed = 0.0
            elif current_risk_consumed > 1.0:
                current_risk_consumed = 1.0
            if current_risk_consumed > max_risk_consumed_short:
                max_risk_consumed_short = current_risk_consumed

            if k == max_hold - 1:
                short_pnl_pct = (entry_price - c_close) / entry_price

        # ---------------- SCORING ----------------
        long_r = long_pnl_pct / risk_pct
        short_r = short_pnl_pct / risk_pct

        if long_r > 0.0:
            long_r *= (1.0 - (mae_penalty * max_risk_consumed_long))
        if short_r > 0.0:
            short_r *= (1.0 - (mae_penalty * max_risk_consumed_short))

        long_r_net = long_r - cost_r
        short_r_net = short_r - cost_r

        if long_r_net > 0.0 and long_r_net > short_r_net:
            targets[i] = np.tanh(long_r_net / saturation_factor)
        elif short_r_net > 0.0 and short_r_net > long_r_net:
            targets[i] = -np.tanh(short_r_net / saturation_factor)

    return targets


def generate_targets(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    max_hold=None,
    fee_per_side=None,
    slippage=None,
    sl_atr_mult=None,
    tp_atr_mult=None,
    enable_trailing=None,
    trail_atr_mult=None,
    saturation_factor=None,
    mae_penalty=None,
):
    """
    Production wrapper for Oracle 5.0.
    Resolves defaults from config and prints target distribution diagnostics.
    """
    if max_hold is None:
        max_hold = config.ORACLE_MAX_HOLD
    if fee_per_side is None:
        fee_per_side = config.FEE_PER_SIDE
    if slippage is None:
        slippage = config.SLIPPAGE
    if sl_atr_mult is None:
        sl_atr_mult = config.ORACLE_SL_ATR_MULT
    if tp_atr_mult is None:
        tp_atr_mult = config.ORACLE_TP_ATR_MULT
    if enable_trailing is None:
        enable_trailing = config.ORACLE_ENABLE_TRAILING
    if trail_atr_mult is None:
        trail_atr_mult = config.ORACLE_TRAIL_ATR_MULT
    if saturation_factor is None:
        saturation_factor = config.SATURATION_FACTOR
    if mae_penalty is None:
        mae_penalty = config.MAE_PENALTY

    targets = _generate_targets_jit(
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        atr_arr=atr_arr,
        max_hold=max_hold,
        fee_per_side=fee_per_side,
        slippage=slippage,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
        enable_trailing=enable_trailing,
        trail_atr_mult=trail_atr_mult,
        saturation_factor=saturation_factor,
        mae_penalty=mae_penalty,
    )

    t = targets
    thresh = getattr(config, "ORACLE_THRESHOLD", getattr(config, "SAMPLER_THRESHOLD", 0.05))
    print(
        f"Target Distribution — Long: {(t > thresh).mean():.3f} | "
        f"Short: {(t < -thresh).mean():.3f} | "
        f"Zero: {(np.abs(t) < thresh).mean():.3f}"
    )
    return targets