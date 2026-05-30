# backtest_engine.py (Production v2)
import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def backtest_one_position(
    signals,
    open_arr, high_arr, low_arr, close_arr, atr_arr,
    first_signal_bar_idx,
    max_hold,
    fee,
    slippage,
    sl_atr_mult=1.5,
    tp_atr_mult=3.0,
    enable_trailing=False,
    trail_atr_mult=1.5,
    cooldown_bars=1,  # 0 = allow re-entry on exit bar; 1 = wait one bar after exit
):
    """
    Backtest engine aligned to windowed-model outputs.

    signals[i] is a decision made at the CLOSE of bar (first_signal_bar_idx + i).
    Entry is executed at that same close price to match oracle label assumption.

    Enforces "one position at a time" by ignoring new signals until the active trade exits.

    Returns:
        pnl_arr: float64[n_signals] PnL per signal index (0.0 if not executed)
        executed_mask: bool_[n_signals] True where a trade was actually entered
        skipped_count: int signals ignored due to position already open
        stopped_early_count: int signals ignored due to insufficient future bars
    """
    n_signals = len(signals)
    pnl_arr = np.zeros(n_signals, dtype=np.float64)
    executed_mask = np.zeros(n_signals, dtype=np.bool_)

    total_cost = (fee + slippage) * 2.0

    in_position = False
    exit_i = -1  # signal-index when we are allowed to evaluate entries again

    skipped_count = 0
    stopped_early_count = 0

    # Past this bar index we cannot simulate full max_hold.
    last_valid_bar_idx = len(close_arr) - max_hold - 1  # inclusive valid max bar index

    for i in range(n_signals):
        # If still in position, we ignore signals until exit_i
        if in_position:
            if i < exit_i:
                if signals[i] != 0:
                    skipped_count += 1
                continue
            in_position = False

        s = signals[i]
        if s == 0:
            continue

        bar_idx = first_signal_bar_idx + i

        # If there's not enough future bars, skip this and keep going (for diagnostics).
        if bar_idx > last_valid_bar_idx:
            stopped_early_count += 1
            continue

        entry_price = close_arr[bar_idx]
        sl_dist = atr_arr[bar_idx] * sl_atr_mult
        tp_dist = atr_arr[bar_idx] * tp_atr_mult
        trail_dist = atr_arr[bar_idx] * trail_atr_mult

        trade_pnl = 0.0
        actual_hold = 0

        if s == 1:  # LONG
            stop_level = entry_price - sl_dist
            tp_level = entry_price + tp_dist
            peak_price = entry_price

            for k in range(1, max_hold):
                j = bar_idx + k
                c_open = open_arr[j]
                c_high = high_arr[j]
                c_low = low_arr[j]
                c_close = close_arr[j]

                # Gap handling
                if c_open <= stop_level:
                    trade_pnl = (c_open - entry_price) / entry_price
                    actual_hold = k
                    break
                if c_open >= tp_level:
                    trade_pnl = (c_open - entry_price) / entry_price
                    actual_hold = k
                    break

                # Trailing stop
                if enable_trailing and c_high > peak_price:
                    peak_price = c_high
                    new_stop = peak_price - trail_dist
                    if new_stop > stop_level:
                        stop_level = new_stop

                # Intrabar resolution
                hit_sl = c_low <= stop_level
                hit_tp = c_high >= tp_level

                if hit_sl and hit_tp:
                    trade_pnl = (stop_level - entry_price) / entry_price
                    actual_hold = k
                    break
                elif hit_sl:
                    trade_pnl = (stop_level - entry_price) / entry_price
                    actual_hold = k
                    break
                elif hit_tp:
                    trade_pnl = (tp_level - entry_price) / entry_price
                    actual_hold = k
                    break

                if k == max_hold - 1:  # time exit
                    trade_pnl = (c_close - entry_price) / entry_price
                    actual_hold = k
                    break

        else:  # SHORT (s == -1)
            stop_level = entry_price + sl_dist
            tp_level = entry_price - tp_dist
            trough_price = entry_price

            for k in range(1, max_hold):
                j = bar_idx + k
                c_open = open_arr[j]
                c_high = high_arr[j]
                c_low = low_arr[j]
                c_close = close_arr[j]

                # Gap handling
                if c_open >= stop_level:
                    trade_pnl = (entry_price - c_open) / entry_price
                    actual_hold = k
                    break
                if c_open <= tp_level:
                    trade_pnl = (entry_price - c_open) / entry_price
                    actual_hold = k
                    break

                # Trailing stop
                if enable_trailing and c_low < trough_price:
                    trough_price = c_low
                    new_stop = trough_price + trail_dist
                    if new_stop < stop_level:
                        stop_level = new_stop

                # Intrabar resolution
                hit_sl = c_high >= stop_level
                hit_tp = c_low <= tp_level

                if hit_sl and hit_tp:
                    trade_pnl = (entry_price - stop_level) / entry_price
                    actual_hold = k
                    break
                elif hit_sl:
                    trade_pnl = (entry_price - stop_level) / entry_price
                    actual_hold = k
                    break
                elif hit_tp:
                    trade_pnl = (entry_price - tp_level) / entry_price
                    actual_hold = k
                    break

                if k == max_hold - 1:  # time exit
                    trade_pnl = (entry_price - c_close) / entry_price
                    actual_hold = k

        pnl_arr[i] = trade_pnl - total_cost
        executed_mask[i] = True
        in_position = True

        # Exit index (signal space): hold bars + optional cooldown
        exit_i = i + actual_hold + cooldown_bars

    return pnl_arr, executed_mask, skipped_count, stopped_early_count
