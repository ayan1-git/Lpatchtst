import numpy as np

data = np.array([1.0, 2.0, 3.0, np.nan])
# np.nanstd default
std_default = np.nanstd(data)
print(f"np.nanstd(data) [default ddof=0]: {std_default}")

# np.nanstd with ddof=1
std_ddof1 = np.nanstd(data, ddof=1)
print(f"np.nanstd(data, ddof=1): {std_ddof1}")

# Manual calculation with ddof=0
mean = np.nanmean(data)
diffs = (data - mean)**2
sum_sq_diffs = np.nansum(diffs)
count = np.count_nonzero(~np.isnan(data))
std_manual_0 = np.sqrt(sum_sq_diffs / count)
print(f"Manual ddof=0: {std_manual_0}")

# Manual calculation with ddof=1
std_manual_1 = np.sqrt(sum_sq_diffs / (count - 1))
print(f"Manual ddof=1: {std_manual_1}")
