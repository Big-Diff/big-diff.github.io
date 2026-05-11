#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert CVRP *.npz (zipped) into memmap directory aligned with
diffusion/co_datasets/memmap_dataset.py (CVRPMemmapVehNodeDataset).

Outputs:
  locs.npy              (M,V,2) float32
  demand_linehaul.npy   (M,N)   float32 (customers only)
  vehicle_capacity.npy  (M,1)   float32
  speed.npy             (M,1)   float32
  best_tour.npy         (M,L)   int32   (0 separators + 0 padding)
  best_cost.npy         (M,)    float32 (nan if missing)
  meta.json             {"written": M, "max_vehicles_seen": ..., ...}
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np


def _pick_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None


def _as_float32(a):
    return np.asarray(a, dtype=np.float32)


def _as_int64(a):
    return np.asarray(a, dtype=np.int64)


def _count_routes_from_actions(actions_1d):
    """Count non-empty segments separated by 0 (0 is depot/sep)."""
    a = np.asarray(actions_1d, dtype=np.int64).reshape(-1)
    # strip trailing zeros only
    end = a.size
    while end > 0 and a[end - 1] == 0:
        end -= 1
    a = a[: max(end, 1)]
    routes = 0
    cur = 0
    for x in a.tolist():
        if x == 0:
            if cur > 0:
                routes += 1
                cur = 0
        else:
            cur += 1
    if cur > 0:
        routes += 1
    return routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--speed", type=float, default=1.0, help="fallback speed if missing")
    ap.add_argument("--max_instances", type=int, default=None)
    args = ap.parse_args()

    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print("[NPZ] keys:", keys)

    # ---- locate arrays (robust to naming) ----
    k_locs = _pick_key(data, ["locs", "locations", "points", "coords", "xy"])
    k_dem  = _pick_key(data, ["demand", "demands", "demand_linehaul", "dem", "d"])
    k_cap  = _pick_key(data, ["cap", "capacity", "vehicle_capacity", "c"])
    k_act  = _pick_key(data, ["actions", "tour", "tours", "best_tour", "route", "routes"])
    k_cost = _pick_key(data, ["cost", "costs", "gt_cost", "best_cost", "obj"])

    if k_locs is None or k_dem is None or k_cap is None or k_act is None:
        raise KeyError(f"Cannot find required keys. Found: {keys}. "
                       f"Need locs/demand/cap/actions at least. "
                       f"Guessed: locs={k_locs}, dem={k_dem}, cap={k_cap}, act={k_act}, cost={k_cost}")

    locs = _as_float32(data[k_locs])          # (M,V,2) expected
    dem  = _as_float32(data[k_dem])           # (M,V) or (M,N)
    cap  = _as_float32(data[k_cap])           # (M,) or (M,1)
    act  = data[k_act]                        # can be (M,L) or object array
    cost = _as_float32(data[k_cost]) if k_cost is not None else None

    M = int(locs.shape[0])
    V = int(locs.shape[1])
    if locs.ndim != 3 or locs.shape[2] != 2:
        raise ValueError(f"locs shape must be (M,V,2), got {locs.shape}")

    if args.max_instances is not None:
        M = min(M, int(args.max_instances))
        locs = locs[:M]
        dem  = dem[:M]
        cap  = cap[:M]
        act  = act[:M]
        if cost is not None:
            cost = cost[:M]

    # demand: keep customers only (N=V-1)
    N = V - 1
    if dem.ndim == 2 and dem.shape[1] == V:
        dem_out = dem[:, 1:].astype(np.float32)
    elif dem.ndim == 2 and dem.shape[1] == N:
        dem_out = dem.astype(np.float32)
    else:
        raise ValueError(f"demand shape mismatch: got {dem.shape}, expected (M,V) or (M,N) with V={V} N={N}")

    # capacity: (M,1)
    cap = cap.reshape(M, -1)
    if cap.shape[1] != 1:
        cap = cap[:, :1]
    cap_out = cap.astype(np.float32)

    # speed: (M,1) constant unless exists in npz
    k_spd = _pick_key(data, ["speed", "spd", "vehicle_speed"])
    if k_spd is not None:
        spd = _as_float32(data[k_spd])[:M].reshape(M, -1)
        if spd.shape[1] != 1:
            spd = spd[:, :1]
        spd_out = spd.astype(np.float32)
    else:
        spd_out = (np.ones((M, 1), dtype=np.float32) * float(args.speed))

    # actions/tours: pad to (M,L) int32
    # handle both fixed 2D array and object array
    if isinstance(act, np.ndarray) and act.dtype != object and act.ndim == 2:
        tour_raw = _as_int64(act)
        L = int(tour_raw.shape[1])
        tour_out = tour_raw.astype(np.int32)
    else:
        # variable length
        lengths = []
        for i in range(M):
            a = _as_int64(act[i]).reshape(-1)
            lengths.append(int(a.size))
        L = int(max(lengths)) if lengths else 1
        tour_out = np.zeros((M, L), dtype=np.int32)
        for i in range(M):
            a = _as_int64(act[i]).reshape(-1)
            l = min(int(a.size), L)
            tour_out[i, :l] = a[:l].astype(np.int32)

    # cost: (M,)
    if cost is None:
        cost_out = np.full((M,), np.nan, dtype=np.float32)
    else:
        cost_out = cost.reshape(-1)[:M].astype(np.float32)
        if cost_out.shape[0] != M:
            cost_out = np.resize(cost_out, (M,)).astype(np.float32)

    # max_vehicles_seen from actions (count routes)
    max_k = 0
    for i in range(M):
        max_k = max(max_k, _count_routes_from_actions(tour_out[i]))

    # ---- write npy (memmap-friendly) ----
    np.save(out_dir / "locs.npy", locs.astype(np.float32))
    np.save(out_dir / "demand_linehaul.npy", dem_out.astype(np.float32))
    np.save(out_dir / "vehicle_capacity.npy", cap_out.astype(np.float32))
    np.save(out_dir / "speed.npy", spd_out.astype(np.float32))
    np.save(out_dir / "best_tour.npy", tour_out.astype(np.int32))
    np.save(out_dir / "best_cost.npy", cost_out.astype(np.float32))

    meta = {
        "written": int(M),
        "V": int(V),
        "N": int(N),
        "L": int(L),
        "max_vehicles_seen": int(max_k),
        "source_npz": str(npz_path),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] wrote memmap dir:", str(out_dir))
    print("[META]", meta)


if __name__ == "__main__":
    main()