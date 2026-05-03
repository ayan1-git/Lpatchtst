import numpy as np
import numba

@numba.jit(nopython=True, cache=True, fastmath=True)
def generate_targets(
    open_arr, high_arr, low_arr, close_arr, atr_arr,
    max_hold, 
    fee_per_side=0.001, 
    slippage=0.0005, 
    atr_mult=3.0, 
    saturation_factor=2.5,
    mae_penalty=0.20
):
    """
    Oracle 4.1: Risk-Standardized & Path-Dependent Targets.
    Fixes vs 4.0:
      - Intrabar trail-stop re-check on both long and short loops
      - MAE penalty now measures proximity to current trailing stop (not entry drawdown)
    """
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    total_cost_pct = (fee_per_side + slippage) * 2.0
    
    stop_distances = atr_arr * atr_mult
    min_vol_pct = 0.001

    for i in range(n - max_hold):
        entry_price = close_arr[i]
        vol_dist = stop_distances[i]
        
        if vol_dist <= 0 or entry_price <= 0: 
            continue
        
        vol_pct = max(vol_dist / entry_price, 0.0001)  # safety floor only — prevents /0

        # Skip structurally untradeable regimes (costs > 1R means no positive EV possible)
        if total_cost_pct / vol_pct > 1.0:
            continue

        # ---------------- LONG LOGIC ----------------
        stop_level = entry_price - vol_dist
        peak_price = entry_price
        max_risk_consumed = 0.0 
        long_pnl_pct = 0.0
        
        for k in range(1, max_hold):
            idx = i + k
            c_open  = open_arr[idx]
            c_low   = low_arr[idx]
            c_high  = high_arr[idx]
            c_close = close_arr[idx]

            # 1. Check Stop Hit (gap or intrabar)
            if c_low <= stop_level:
                exit_price = min(c_open, stop_level)
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed = 1.0
                break
            
            # 2. Trail Stop + intrabar re-check
            if c_high > peak_price:
                peak_price = c_high
                new_stop = peak_price - vol_dist
                if new_stop > stop_level:
                    stop_level = new_stop
                    if c_low <= stop_level:
                        exit_price = stop_level
                        long_pnl_pct = (exit_price - entry_price) / entry_price
                        max_risk_consumed = 1.0
                        break
            
            # 3. Update Risk Consumption (proximity to current trailing stop)
            current_risk_consumed = min(1.0, max(0.0, (vol_dist - (c_low - stop_level)) / vol_dist))
            if current_risk_consumed > max_risk_consumed:
                max_risk_consumed = current_risk_consumed
            
            # 4. Time Exit
            if k == max_hold - 1:
                long_pnl_pct = (c_close - entry_price) / entry_price

        # ---------------- SHORT LOGIC ----------------
        stop_level_short = entry_price + vol_dist
        trough_price = entry_price
        max_risk_consumed_short = 0.0
        short_pnl_pct = 0.0
        
        for k in range(1, max_hold):
            idx = i + k
            c_open  = open_arr[idx]
            c_high  = high_arr[idx]
            c_low   = low_arr[idx]
            c_close = close_arr[idx]

            # 1. Check Stop Hit
            if c_high >= stop_level_short:
                exit_price = max(c_open, stop_level_short)
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                break
            
            # 2. Trail Stop + intrabar re-check
            if c_low < trough_price:
                trough_price = c_low
                new_stop = trough_price + vol_dist
                if new_stop < stop_level_short:
                    stop_level_short = new_stop
                    if c_high >= stop_level_short:
                        exit_price = stop_level_short
                        short_pnl_pct = (entry_price - exit_price) / entry_price
                        max_risk_consumed_short = 1.0
                        break
            
            # 3. Update Risk Consumption (proximity to current trailing stop)
            current_risk_consumed = min(1.0, max(0.0, (vol_dist - (stop_level_short - c_high)) / vol_dist))
            if current_risk_consumed > max_risk_consumed_short:
                max_risk_consumed_short = current_risk_consumed
            
            # 4. Time Exit
            if k == max_hold - 1:
                short_pnl_pct = (entry_price - c_close) / entry_price

        # ---------------- SCORING ----------------
        long_r  = long_pnl_pct  / vol_pct
        short_r = short_pnl_pct / vol_pct
        
        max_risk_consumed       = min(max(max_risk_consumed, 0.0), 1.0)
        max_risk_consumed_short = min(max(max_risk_consumed_short, 0.0), 1.0)
        
        if long_r > 0:
            long_r *= (1.0 - (mae_penalty * max_risk_consumed))
        if short_r > 0:
            short_r *= (1.0 - (mae_penalty * max_risk_consumed_short))
            
        cost_r = total_cost_pct / vol_pct
        long_r_net  = long_r  - cost_r
        short_r_net = short_r - cost_r
        
        if long_r_net > 0 and long_r_net > short_r_net:
            targets[i] = np.tanh(long_r_net / saturation_factor)
        elif short_r_net > 0 and short_r_net > long_r_net:
            targets[i] = -np.tanh(short_r_net / saturation_factor)
            
    return targets
