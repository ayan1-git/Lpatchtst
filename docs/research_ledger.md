### Run 00: Baseline
- **Δ Code:** N/A (Initial baseline).
- **Outcome:** PF: 0.289; Net Ret: -25.78%; Win Rate: 26.0%; Test Std Dev: 0.685 (Target: 0.400).
- **Verdict:** PIVOT.

## 2026-05-30 10:00
### Run 01: Reward & Loss Tuning
- **Δ Code:** `loss.py`: Quantiles 0.40/0.60; `dir_reward`: magnitude gate added; `spread_reward`: real_signal threshold 0.10; `global_std_floor`: 0.35 -> 0.20.
- **Outcome:** Val Std: 0.25 (Target: 0.31); Test Std: 0.16 (Target: 0.40); Net Return: -0.0155; Profit Factor: 0.771; Win Rate: 38.9%.
- **Verdict:** KEEP.

## 2026-05-30 14:30
### Run 02: Per-Fold Bias Reset
- **Δ Code:** `train.py`: `model.head` weights $\rightarrow$ `normal_(0, 0.01)`, bias $\rightarrow$ `zero_()`.
- **Outcome:** Only Fold 1 saved; Val Loss $\uparrow$ in subsequent folds.
- **Verdict:** REVERT.
