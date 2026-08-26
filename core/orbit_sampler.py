# orbit_sampler.py
#
# Bootstrap Multi-Level Sampling and low-discrepancy greedy blending from
# "Into the ORBIT for Time Series: Training Regimes for Foundation Models"
# (arXiv:2608.13262, Falcon-2.0 technical report), adapted to this repo's
# multi-asset pretraining pipeline.
#
# Paper mapping
# -------------
# Dataset        := one asset CSV (exposure controlled by prescribed weights)
# Record  (L-1)  := the asset's single series            (identity degenerate)
# Variable (L-2) := the single oracle-scored target      (identity degenerate)
# Window   (L-3) := context [s, e) drawn per paper Eq.31–32 analog:
#                   e ~ Uniform{ctx_min .. L - p_max}
#                   s ~ Uniform{max(0, e - C) .. e - ctx_min}
# Horizon  (L-4) := p ~ Uniform over feasible ORBIT_HORIZONS entries with
#                   label bar e-1 + p <= L-1.
#
# The index is built OFFLINE once (seeded, cacheable) exactly as in the paper:
# contexts/horizons are sampled at construction time and never resampled
# during optimization.

from __future__ import annotations

import hashlib
import heapq
import json
import os

import numpy as np


def state_hash(asset_ids: list[str], lengths: list[int], horizons: list[int],
               ctx_min: int, max_ctx: int, seed: int) -> str:
    """Deterministic key binding the cached index to its generating config."""
    payload = json.dumps(
        {
            "assets": list(map(str, asset_ids)),
            "lengths": [int(x) for x in lengths],
            "horizons": sorted(int(h) for h in horizons),
            "ctx_min": int(ctx_min),
            "max_ctx": int(max_ctx),
            "seed": int(seed),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def build_bootstrap_index(
    lengths: list[int],
    horizons: list[int],
    alloc_counts: list[int],
    ctx_min: int,
    max_ctx: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Paper Algorithm 1 (levels 3–4) per dataset.

    Returns one int64 array of shape (M_d, 3) per dataset with columns
    [context_start s, context_end e, horizon_index]. Validity is guaranteed
    by construction:
        ctx_min <= e <= L - p_max          (label bar e-1 + h <= L-1 for ALL h)
        max(0, e-C) <= s <= e - ctx_min     (ctx length >= ctx_min, <= C)
    """
    horizons_arr = np.asarray(sorted(horizons), dtype=np.int64)
    n_h = len(horizons_arr)
    index: list[np.ndarray] = []

    for L, m in zip(lengths, alloc_counts):
        m = int(m)
        if L < ctx_min + int(horizons_arr.max()) or m <= 0:
            index.append(np.zeros((0, 3), dtype=np.int64))
            continue

        e_max = int(L) - int(horizons_arr.max())
        e = rng.integers(ctx_min, e_max + 1, size=m)

        s_lo = np.maximum(0, e - int(max_ctx))
        span = (e - ctx_min) - s_lo + 1
        if (span <= 0).any():
            raise ValueError(
                f"Bootstrap sampling produced an empty context range "
                f"(L={L}, ctx_min={ctx_min}, max_ctx={max_ctx})."
            )
        s = s_lo + rng.integers(0, span, size=m)

        h_idx = rng.integers(0, n_h, size=m)

        # Guaranteed by e <= L - p_max; kept as a hard guard.
        assert (e - 1 + horizons_arr[h_idx] <= np.int64(L) - 1).all(), \
            "infeasible (e, h) pair escaped validity clamp"
        assert (e - s >= ctx_min).all() and (s >= 0).all()

        entry = np.stack([s, e, h_idx], axis=1).astype(np.int64)
        index.append(entry)

    return index


def greedy_blend(weights: list[float], total_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Low-discrepancy greedy blending rule (paper §4.2.1).

    For each slot j the rule selects the dataset whose cumulative assigned
    count has the largest deficit relative to its target count:
        argmax_d ( w_d * j - c_d )   ≡   argmin_d c_d / w_d
    so cumulative composition tracks the prescribed weights after EVERY
    assignment (unlike categorical sampling which matches them only in
    expectation).

    Returns
    -------
    slot_dataset : int32[total_slots] — dataset id owning each stream slot
    slot_rank    : int64[total_slots] — occurrence rank of the slot within
                   its dataset (k-th occurrence gets local sample id k)
    """
    weights = np.asarray(weights, dtype=np.float64)
    if (weights <= 0).any():
        raise ValueError("greedy_blend: all weights must be positive")
    D = len(weights)

    heap = [(0.0, d) for d in range(D)]  # (fill ratio c_d/w_d, d) — tie-break by d
    heapq.heapify(heap)
    counts = np.zeros(D, dtype=np.int64)

    slot_dataset = np.empty(total_slots, dtype=np.int32)
    slot_rank = np.empty(total_slots, dtype=np.int64)
    next_rank = np.zeros(D, dtype=np.int64)

    for j in range(total_slots):
        ratio, d = heapq.heappop(heap)
        # `d` is a dataset id (int). Keep it as int — float would break
        # numpy indexing on next_rank / counts / slot_dataset below.
        slot_dataset[j] = d
        slot_rank[j] = next_rank[d]
        next_rank[d] += 1
        counts[d] += 1
        heapq.heappush(heap, (counts[d] / weights[d], d))

    return slot_dataset, slot_rank


def build_orbit_stream(
    index: list[np.ndarray],
    weights: list[float],
    total_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Assemble the global training stream (paper §4.2.1 + §4.3).

    Two-level shuffle before consumption: sample identifiers within each
    dataset are shuffled independently (the per-dataset permutation), then the
    globally interleaved slot order comes from the greedy blender itself.

    Resolution of stream slot j:
        d = slot_dataset[j]; local_id = perm[d][slot_rank[j]]
        (s, e, h_idx) = index[d][local_id]
    """
    slot_dataset, slot_rank = greedy_blend(weights, total_samples)
    perm = [rng.permutation(len(ix)) if len(ix) else np.zeros(0, dtype=np.int64)
            for ix in index]
    return {
        "slot_dataset": slot_dataset,
        "slot_rank": slot_rank,
        "perm": [p.astype(np.int64) for p in perm],
    }


def allocation_counts(weights: list[float], total_samples: int) -> list[int]:
    """Per-dataset offline index sizes covering every blended slot.

    Uses largest-remainder rounding so sum(M_d) == total_samples and each
    M_d >= its exact share (no stream slot ever exhausts its dataset's index).
    """
    w = np.asarray(weights, dtype=np.float64)
    shares = w / w.sum() * total_samples
    floors = np.floor(shares).astype(np.int64)
    remainder = int(total_samples - floors.sum())
    # Distribute leftovers to the largest fractional parts.
    frac_order = np.argsort(-(shares - floors))
    floors[frac_order[:remainder]] += 1
    return [int(x) for x in floors]


def save_index_cache(path: str, h: str, index: list[np.ndarray]) -> None:
    """Atomically persist the offline index under its config hash."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    payload = {"hash": np.array([h])}
    for i, ix in enumerate(index):
        payload[f"idx{i}"] = ix
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def load_index_cache(path: str, h: str) -> list[np.ndarray] | None:
    """Load a cached index if it exists AND was built for this exact config."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            stored = str(z["hash"][0])
            if stored != h:
                return None
            keys = sorted(
                (k for k in z.files if k.startswith("idx")),
                key=lambda k: int(k[3:]),
            )
            return [z[k].astype(np.int64) for k in keys]
    except Exception:
        return None
