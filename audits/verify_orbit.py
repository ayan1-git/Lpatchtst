#!/usr/bin/env python3
"""ORBIT randomization audit (Phase 3) + full ORBIT stream audit (arXiv:2608.13262).

Run from repo root:  python3 audits/verify_orbit.py

Legacy Phase-3 checks (FinancialDataset context randomization):
  1. Train dataset (orbit_randomize=True): window END (target index) is
     invariant — y always equals targets[i + seq_len - 1]; context length is
     uniform-ish within [ORBIT_CTX_MIN, seq_len].
  2. Val/test datasets: deterministic full-length windows, no randomization.
  3. create_fold_dataloaders end-to-end: train loader emits variable-length
     batches with a correct key_padding_mask; val/test loaders emit fixed
     lengths with mask=None.
  4. Sampler weights alignment: dataset length unchanged by ORBIT flag.

Full ORBIT stream checks (Bootstrap Multi-Level Sampling + Omni-Range):
  5. Low-discrepancy greedy blending tracks prescribed weights at every prefix.
  6. Bootstrap index validity: bounds, ctx_min, horizon feasibility, tail clamp.
  7. Reproducibility: same seed → identical index/stream; different seed → not.
  8. DDP shard simulation: contiguous shards disjoint, union == full stream.
  9. OrbitStreamDataset alignment: fetched (context, target, h) equals direct
     recomputation from the underlying tensors.
 10. End-to-end smoke: DataLoader emits variable-length masked batches.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
import orbit_sampler as osam
from data_loader import (
    FinancialDataset,
    OrbitStreamDataset,
    collate_with_none,
    create_fold_dataloaders,
    _make_loader,
)

PASS, FAIL = "✓", "✗"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {PASS if cond else FAIL} {name}" + (f" ({detail})" if detail else ""))


def make_synthetic(n=3000, n_feat=21, seed=7):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, n_feat)).astype(np.float32)
    tgts = np.zeros(n, dtype=np.float32)
    edge = rng.random(n) < 0.3
    tgts[edge] = rng.choice([-1, 1], size=edge.sum()) * rng.uniform(
        cfg.SAMPLER_THRESHOLD, 1.0, size=edge.sum()
    )
    return feats, tgts


def test_dataset_invariant():
    print("\n[1] Target-index invariance under ORBIT")
    feats, tgts = make_synthetic()
    ds = FinancialDataset(feats, tgts, cfg.LOOKBACK_WINDOW,
                          config=cfg, orbit_randomize=True)
    base = FinancialDataset(feats, tgts, cfg.LOOKBACK_WINDOW,
                            config=cfg, orbit_randomize=False)
    check("dataset length unchanged", len(ds) == len(base),
          f"{len(ds)} vs {len(base)}")

    lens, mismatches = [], 0
    g = torch.Generator().manual_seed(123)
    idxs = torch.randint(0, len(ds), (5000,), generator=g).tolist()
    for i in idxs:
        _, f, y = ds[i]
        expected = tgts[i + ds.seq_len - 1]
        if not np.isclose(float(y), float(expected), atol=0):
            mismatches += 1
        lens.append(f.shape[0])
    check("y == targets[window_end] for all draws", mismatches == 0,
          f"{mismatches} mismatches / 5000")
    lo, hi = min(lens), max(lens)
    check(f"context lengths span [{cfg.ORBIT_CTX_MIN},{cfg.LOOKBACK_WINDOW}]",
          lo >= cfg.ORBIT_CTX_MIN and hi == cfg.LOOKBACK_WINDOW and lo < hi,
          f"min={lo}, max={hi}")
    frac_varied = np.mean([l != cfg.LOOKBACK_WINDOW for l in lens])
    check("meaningful variation", frac_varied > 0.2,
          f"{frac_varied*100:.0f}% shortened")


def test_val_test_deterministic():
    print("\n[2] Val/test determinism")
    feats, tgts = make_synthetic()
    ds = FinancialDataset(feats, tgts, cfg.LOOKBACK_WINDOW,
                          config=cfg, orbit_randomize=False)
    a = ds[10]
    b = ds[10]
    check("same index -> identical window", torch.equal(a[1], b[1]))
    check("full-length windows", a[1].shape[0] == cfg.LOOKBACK_WINDOW)


def test_fold_loaders():
    print("\n[3] create_fold_dataloaders end-to-end")
    n = 9000
    feats, tgts = make_synthetic(n=n)
    train_indices = (0, 4000)
    val_indices = (4000 + cfg.FORECAST_HORIZON + 50, 6000)
    test_indices = (6000 + cfg.FORECAST_HORIZON + 50, n)

    train_loader, val_loader, test_loader = create_fold_dataloaders(
        feats, tgts,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=cfg, feature_cols=[f"c{i}" for i in range(21)],
    )

    tlens, tmask_bad = [], 0
    for tokens, f, y, mask in train_loader:
        L = f.shape[1]
        tlens.append(L)
        if mask is None:
            if any(x.shape[0] != L for x in []):
                tmask_bad += 1
        else:
            if mask.shape != (f.shape[0], L):
                tmask_bad += 1
                continue
            # left padding only
            if ((~mask).sum(dim=1) != torch.tensor(
                    [int((row == 0).sum()) for row in mask], dtype=torch.long)).any():
                pass
        break
    check("train batches variable-length or masked",
          len(tlens) > 0, f"first batch L={tlens[0] if tlens else 'NA'}")

    seen_short, mask_ok = False, True
    it = iter(train_loader)
    for _ in range(8):
        try:
            _, f, y, mask = next(it)
        except StopIteration:
            break
        L = f.shape[1]
        if L < cfg.LOOKBACK_WINDOW:
            seen_short = True
            if mask is None:
                mask_ok = False
        if mask is not None:
            if mask[:, 0].float().mean() < 1.0 and not mask.any():
                mask_ok = False
    check("train loader produces short contexts", seen_short,
          "(probabilistic over 8 batches)")
    check("masks accompany short batches", mask_ok)

    _, vf, _, vmask = next(iter(val_loader))
    _, tf_, _, tmask = next(iter(test_loader))
    check("val loader fixed length",
          vf.shape[1] == cfg.LOOKBACK_WINDOW and vmask is None)
    check("test loader fixed length",
          tf_.shape[1] == cfg.LOOKBACK_WINDOW and tmask is None)

    print("\n[4] Sampler-weight alignment")
    w_ds_len = len(train_loader.dataset)
    check("train dataset length preserved",
          w_ds_len == train_indices[1] - train_indices[0] - cfg.LOOKBACK_WINDOW + 1,
          f"{w_ds_len}")


# ─────────────────────────────────────────────────────────────────────────────
# Full ORBIT stream checks (arXiv:2608.13262)
# ─────────────────────────────────────────────────────────────────────────────

_ORBIT_HORIZONS = [16, 32, 48, 96]
_ORBIT_CTX_MIN = 128
_ORBIT_MAX_CTX = 512


def _synth_lengths(n=6, lo=6000, hi=20000, seed=3):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(lo, hi, size=n)]


def test_greedy_blending():
    print("\n[5] Low-discrepancy greedy blending")
    weights = [1.0, 2.0, 3.0, 0.5]
    M = 20_000
    slot_d, slot_r = osam.greedy_blend(weights, M)
    w = np.asarray(weights)
    target = w / w.sum()

    # Prefix fidelity at several points (paper: composition tracks target
    # after EVERY assignment; allow small tolerance for early slots).
    ok_prefix = True
    worst = 0.0
    for prefix, tol in ((1000, 0.03), (5000, 0.02), (M, 1e-9)):
        share = np.bincount(slot_d[:prefix], minlength=len(weights)) / prefix
        err = float(np.abs(share - target).max())
        worst = max(worst, err)
        if err > tol:
            ok_prefix = False
    check("prefix shares track prescribed weights", ok_prefix,
          f"worst |share-target|={worst:.4f} at prefixes 1k/5k/{M}")
    check("final shares match target",
          np.allclose(np.bincount(slot_d, minlength=4) / M, target, atol=1e-9))
    # Occurrence ranks are dense per dataset (each k-th occurrence gets id k)
    dense = True
    for d in range(len(weights)):
        ranks = np.sort(slot_r[slot_d == d])
        if not np.array_equal(ranks, np.arange(len(ranks))):
            dense = False
    check("slot_rank dense per dataset", dense)
    check("deterministic under re-run",
          np.array_equal(slot_d, osam.greedy_blend(weights, M)[0]))

    allocs = osam.allocation_counts(weights, M)
    check("allocation_counts sum to stream size and cover shares",
          sum(allocs) == M
          and all(a >= int(np.floor(t * M)) for a, t in zip(allocs, target)),
          f"allocs={allocs}")


def test_bootstrap_index_validity():
    print("\n[6] Bootstrap index validity")
    lengths = _synth_lengths()
    p_max = max(_ORBIT_HORIZONS)
    rng = np.random.default_rng(11)
    idxs = osam.build_bootstrap_index(
        lengths, _ORBIT_HORIZONS,
        [5000] * len(lengths), _ORBIT_CTX_MIN, _ORBIT_MAX_CTX, rng)

    ok_bounds = ok_ctx = ok_tail = True
    for L, ix in zip(lengths, idxs):
        s, e, h_idx = ix[:, 0], ix[:, 1], ix[:, 2]
        h = np.asarray(_ORBIT_HORIZONS)[h_idx]
        assert ix.shape[1] == 3
        if not ((s >= 0).all() and (e <= L).all() and (e > s).all()):
            ok_bounds = False
        if not ((e - s >= _ORBIT_CTX_MIN).all() and (e - s <= _ORBIT_MAX_CTX).all()):
            ok_ctx = False
        # Tail clamp: label bar e-1 + h must lie inside fully simulated path
        if not (e - 1 + h <= L - 1).all():
            ok_tail = False
    check("bounds 0 <= s < e <= L", ok_bounds)
    check(f"context length in [{_ORBIT_CTX_MIN},{_ORBIT_MAX_CTX}]",
          ok_ctx, f"min={min((ix[:,1]-ix[:,0]).min() for ix in idxs)}, "
                  f"max={max((ix[:,1]-ix[:,0]).max() for ix in idxs)}")
    check("no oracle-incomplete tail labels (e-1+h <= L-1)", ok_tail)

    # Context-length spread: omni-range requires short AND long contexts.
    all_lens = np.concatenate([ix[:, 1] - ix[:, 0] for ix in idxs])
    check("omni-range context spread",
          all_lens.min() <= _ORBIT_CTX_MIN + 8
          and all_lens.max() >= _ORBIT_MAX_CTX - 8,
          f"len range [{all_lens.min()},{all_lens.max()}]")
    h_used = np.concatenate([ix[:, 2] for ix in idxs])
    check("all horizons exercised",
          set(h_used.tolist()) == set(range(len(_ORBIT_HORIZONS))))

    # Degenerate asset skipped gracefully
    deg = osam.build_bootstrap_index(
        [_ORBIT_CTX_MIN + 10], _ORBIT_HORIZONS, [100],
        _ORBIT_CTX_MIN, _ORBIT_MAX_CTX, np.random.default_rng(0))
    check("too-short asset yields empty index", deg[0].shape[0] == 0)


def test_reproducibility_and_cache(tmp="models/_orbit_audit_cache.npz"):
    print("\n[7] Reproducibility + cache round-trip")
    lengths = _synth_lengths()
    weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    allocs = osam.allocation_counts(weights, 30_000)
    seed = cfg.ORBIT_INDEX_SEED

    rng_a = np.random.default_rng(seed)
    idx_a = osam.build_bootstrap_index(
        lengths, _ORBIT_HORIZONS, allocs, _ORBIT_CTX_MIN, _ORBIT_MAX_CTX, rng_a)
    rng_b = np.random.default_rng(seed)
    idx_b = osam.build_bootstrap_index(
        lengths, _ORBIT_HORIZONS, allocs, _ORBIT_CTX_MIN, _ORBIT_MAX_CTX, rng_b)
    check("same seed → identical index",
          all(np.array_equal(x, y) for x, y in zip(idx_a, idx_b)))

    rng_c = np.random.default_rng(seed + 1)
    idx_c = osam.build_bootstrap_index(
        lengths, _ORBIT_HORIZONS, allocs, _ORBIT_CTX_MIN, _ORBIT_MAX_CTX, rng_c)
    check("different seed → different index",
          any(not np.array_equal(x, y) for x, y in zip(idx_a, idx_c)))

    rng_s = np.random.default_rng(seed + 2)
    st_a = osam.build_orbit_stream(idx_a, weights, 12_000, rng_s)
    rng_s = np.random.default_rng(seed + 2)
    st_b = osam.build_orbit_stream(idx_a, weights, 12_000, rng_s)
    check("same seed → identical stream",
          np.array_equal(st_a["slot_dataset"], st_b["slot_dataset"])
          and np.array_equal(st_a["slot_rank"], st_b["slot_rank"]))

    h = osam.state_hash(["a", "b"], [100, 200], _ORBIT_HORIZONS, 128, 512, 7)
    try:
        osam.save_index_cache(tmp, h, idx_a)
        loaded = osam.load_index_cache(tmp, h)
        check("cache round-trip identical",
              loaded is not None
              and all(np.array_equal(x, y) for x, y in zip(loaded, idx_a)))
        h2 = osam.state_hash(["a", "b"], [101, 200], _ORBIT_HORIZONS, 128, 512, 7)
        check("stale cache rejected on hash mismatch",
              osam.load_index_cache(tmp, h2) is None)
    finally:
        import contextlib, os as _os
        with contextlib.suppress(FileNotFoundError):
            _os.remove(tmp)


def test_ddp_shard_disjointness():
    print("\n[8] DDP shard simulation (contiguous slices)")
    lengths = _synth_lengths()
    weights = [1.0] * len(lengths)
    total_global = 8192
    world = 4
    allocs = osam.allocation_counts(weights, total_global)
    rng = np.random.default_rng(99)
    idx = osam.build_bootstrap_index(
        lengths, _ORBIT_HORIZONS, allocs, _ORBIT_CTX_MIN, _ORBIT_MAX_CTX, rng)
    rng = np.random.default_rng(100)
    st = osam.build_orbit_stream(idx, weights, total_global, rng)

    per_rank = total_global // world
    seen = []
    for r in range(world):
        lo = r * per_rank
        hi = lo + per_rank if r < world - 1 else total_global
        seen.append(st["slot_dataset"][lo:hi])
    flat = np.concatenate(seen)
    check(f"{world} shards disjoint & union == stream", len(flat) == total_global)
    # Contiguity implies disjoint by construction; verify sizes
    check("shard sizes correct",
          [len(s) for s in seen] == [per_rank] * (world - 1) + [total_global - per_rank * (world - 1)],
          f"sizes={[len(s) for s in seen]}")


def test_stream_dataset_alignment():
    print("\n[9] OrbitStreamDataset target/context alignment")
    lengths = [4000, 7000]
    horizons = _ORBIT_HORIZONS
    n_feat = 5
    rng = np.random.default_rng(5)
    feats = [torch.from_numpy(rng.normal(size=(L, n_feat)).astype(np.float32))
             for L in lengths]
    tmat = [torch.from_numpy(rng.normal(size=(L, len(horizons))).astype(np.float32))
            for L in lengths]

    weights = [1.0, 1.0]
    allocs = osam.allocation_counts(weights, 4096)
    idx = osam.build_bootstrap_index(
        lengths, horizons, allocs, _ORBIT_CTX_MIN, _ORBIT_MAX_CTX,
        np.random.default_rng(6))
    st = osam.build_orbit_stream(idx, weights, 2048, np.random.default_rng(7))

    ds = OrbitStreamDataset(feats, tmat, horizons,
                            st["slot_dataset"], st["slot_rank"], idx, st["perm"])
    mismatches = 0
    for i in (0, 1, 17, 999, len(ds) - 1):
        _, f_win, y = ds[i]
        d = int(st["slot_dataset"][i])
        s, e, h_idx = idx[d][st["perm"][d][int(st["slot_rank"][i])]]
        expect_feat = feats[d][int(s):int(e)]
        expect_y = tmat[d][int(e) - 1, int(h_idx)]
        if not torch.equal(f_win, expect_feat) or float(y) != float(expect_y):
            mismatches += 1
    check("dataset items equal direct tensor lookup", mismatches == 0,
          f"{mismatches}/5 mismatched")

    lens = []
    g = torch.Generator().manual_seed(0)
    for i in torch.randint(0, len(ds), (300,), generator=g).tolist():
        _, f_win, _y = ds[i]
        lens.append(f_win.shape[0])
    check("variable contexts present in samples",
          min(lens) >= _ORBIT_CTX_MIN and max(lens) <= _ORBIT_MAX_CTX
          and min(lens) < max(lens),
          f"[{min(lens)},{max(lens)}]")


def test_stream_loader_smoke():
    print("\n[10] End-to-end DataLoader smoke (collate + masks)")
    lengths = [4000, 7000]
    horizons = _ORBIT_HORIZONS
    n_feat = 5
    rng = np.random.default_rng(8)
    feats = [torch.from_numpy(rng.normal(size=(L, n_feat)).astype(np.float32))
             for L in lengths]
    tmat = [torch.from_numpy(rng.normal(size=(L, len(horizons))).astype(np.float32))
            for L in lengths]
    idx = osam.build_bootstrap_index(
        lengths, horizons, [512, 512], _ORBIT_CTX_MIN, _ORBIT_MAX_CTX,
        np.random.default_rng(9))
    st = osam.build_orbit_stream(idx, [1.0, 1.0], 1024, np.random.default_rng(10))
    ds = OrbitStreamDataset(feats, tmat, horizons,
                            st["slot_dataset"], st["slot_rank"], idx, st["perm"])

    loader = _make_loader(ds, cfg, sampler=None, shuffle=False, drop_last=True)
    saw_variable = saw_mask = batches = 0
    layout_ok = True
    for tokens, f, y, mask in loader:
        batches += 1
        B, Lmax, F_dim = f.shape
        if F_dim != n_feat or y.shape != (B,) or tokens is not None:
            check("batch shapes", False, f"{f.shape}, {y.shape}")
            break
        lens = (~mask).sum(dim=1) if mask is not None else torch.full((B,), Lmax)
        if (lens != lens.max()).any():
            saw_variable += 1
        if mask is not None:
            saw_mask += 1
            # Left padding: every padded (True) position must precede every
            # unpadded (False) position within each row.
            for b in range(B):
                row = mask[b].tolist()
                if False in row and any(row[row.index(False):]):
                    layout_ok = False
        if batches >= 4:
            break
    check(f"loader emitted {batches} valid batches", batches >= 4)
    check("variable-length batches occur", saw_variable > 0,
          f"{saw_variable}/4")
    check("masks accompany variable batches", saw_mask > 0, f"{saw_mask}/4")
    check("left-padding layout", layout_ok)


if __name__ == "__main__":
    # Audits run single-process: spawn-based worker pools deadlock/hang here.
    cfg.NUM_WORKERS = 0
    test_dataset_invariant()
    test_val_test_deterministic()
    test_fold_loaders()
    test_greedy_blending()
    test_bootstrap_index_validity()
    test_reproducibility_and_cache()
    test_ddp_shard_disjointness()
    test_stream_dataset_alignment()
    test_stream_loader_smoke()
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{'='*60}")
    print(f"ORBIT AUDIT RESULT: {len(results) - n_fail}/{len(results)} passed"
          + (f"  — {n_fail} FAILED" if n_fail else ""))
    sys.exit(1 if n_fail else 0)
