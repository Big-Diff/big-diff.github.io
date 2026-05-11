import os
import json
import argparse
import random
import numpy as np
import torch

from diffusion.co_datasets.memmap_dataset import CVRPMemmapVehNodeDataset


def _isfinite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _check_tensor(name: str, t: torch.Tensor, *, allow_empty=False, minv=None, maxv=None):
    if t is None:
        return f"[MISS] {name} is None"
    if not torch.is_tensor(t):
        return f"[TYPE] {name} not tensor: {type(t)}"
    if (t.numel() == 0) and (not allow_empty):
        return f"[EMPTY] {name} empty"
    if not _isfinite(t):
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
        return f"[NAN/INF] {name} has non-finite, first_bad_idx={bad[0].tolist() if bad.numel() else '??'}"
    if minv is not None:
        if (t < minv).any().item():
            v = t.min().item()
            return f"[RANGE] {name} has < {minv}, min={v}"
    if maxv is not None:
        if (t > maxv).any().item():
            v = t.max().item()
            return f"[RANGE] {name} has > {maxv}, max={v}"
    return None


def check_one(ds, idx: int):
    g = ds[idx]

    # ---- required fields (by your model contract) ----
    node = getattr(g, "node_features", None)
    veh = getattr(g, "veh_features", None)
    eidx = getattr(g, "edge_index", None)
    eattr = getattr(g, "edge_attr", None)

    # optional but commonly present
    y = getattr(g, "y", None)
    dem = getattr(g, "demand_linehaul", None)
    cap = getattr(g, "capacity", None)
    Ku = getattr(g, "K_used", None)
    depot_xy = getattr(g, "depot_xy", None)

    errs = []

    # tensor checks
    for name, t, kw in [
        ("node_features", node, dict()),
        ("veh_features", veh, dict()),
        ("edge_index", eidx, dict()),
        ("edge_attr", eattr, dict()),
    ]:
        msg = _check_tensor(name, t, **kw)
        if msg:
            errs.append(msg)

    # y / demand / cap
    if y is not None:
        msg = _check_tensor("y", y)
        if msg:
            errs.append(msg)
    if dem is not None:
        msg = _check_tensor("demand_linehaul", dem, minv=0.0)
        if msg:
            errs.append(msg)
    if cap is not None and torch.is_tensor(cap):
        msg = _check_tensor("capacity", cap, minv=0.0)
        if msg:
            errs.append(msg)
    if depot_xy is not None:
        msg = _check_tensor("depot_xy", depot_xy)
        if msg:
            errs.append(msg)

    # shape checks
    if torch.is_tensor(node) and node.dim() == 2:
        N = int(node.size(0))
    else:
        N = None
    if torch.is_tensor(veh) and veh.dim() == 2:
        K = int(veh.size(0))
    else:
        K = None

    if torch.is_tensor(dem) and N is not None:
        if int(dem.numel()) != N:
            errs.append(f"[SHAPE] demand_linehaul numel={int(dem.numel())} != N={N}")

    if torch.is_tensor(y) and N is not None:
        if int(y.numel()) != N:
            errs.append(f"[SHAPE] y numel={int(y.numel())} != N={N}")

    if torch.is_tensor(eidx):
        if eidx.dim() != 2 or eidx.size(0) != 2:
            errs.append(f"[SHAPE] edge_index shape={tuple(eidx.shape)} expected (2,E)")
        else:
            src = eidx[0].long()
            dst = eidx[1].long()

            if (src < 0).any().item() or (dst < 0).any().item():
                errs.append("[RANGE] edge_index has negative indices")

            if K is not None and src.numel() > 0 and int(src.max().item()) >= K:
                errs.append(f"[RANGE] edge_index src max={int(src.max().item())} >= K={K}")
            if N is not None and dst.numel() > 0 and int(dst.max().item()) >= N:
                errs.append(f"[RANGE] edge_index dst max={int(dst.max().item())} >= N={N}")

    if torch.is_tensor(eattr) and torch.is_tensor(eidx):
        if eattr.dim() != 2:
            errs.append(f"[SHAPE] edge_attr shape={tuple(eattr.shape)} expected (E,dim)")
        else:
            E = int(eidx.size(1)) if (eidx.dim() == 2 and eidx.size(0) == 2) else None
            if E is not None and int(eattr.size(0)) != E:
                errs.append(f"[SHAPE] edge_attr E={int(eattr.size(0))} != edge_index E={E}")

    # Ku/y range checks
    if Ku is not None and torch.is_tensor(Ku):
        # Ku could be scalar tensor
        Ku_i = int(Ku.view(-1)[0].item())
        if Ku_i <= 0:
            errs.append(f"[RANGE] K_used={Ku_i} <= 0")
        if K is not None and Ku_i > K:
            errs.append(f"[RANGE] K_used={Ku_i} > K={K}")

        if torch.is_tensor(y):
            yy = y.long().view(-1)
            if (yy < 0).any().item():
                errs.append("[RANGE] y has negative labels")
            if (yy >= Ku_i).any().item():
                errs.append(f"[RANGE] y has labels >= K_used ({Ku_i})")

    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, type=str)
    ap.add_argument("--num", default=2000, type=int, help="How many indices to check")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--K_max", default=0, type=int, help="Override K_max (0 means read meta.json)")
    ap.add_argument("--sparse_factor", default=-1, type=int)
    args = ap.parse_args()

    path = args.path
    meta_path = os.path.join(path, "meta.json")
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        print("[meta.json]", meta)
        k_meta = int(meta.get("max_vehicles_seen", 0) or 0)
    else:
        print("[WARN] meta.json not found:", meta_path)
        meta = {}
        k_meta = 0

    K_max = int(args.K_max) if int(args.K_max) > 0 else int(k_meta)
    if K_max <= 0:
        raise RuntimeError("K_max is not set. Provide --K_max or ensure meta.json has max_vehicles_seen.")

    ds = CVRPMemmapVehNodeDataset(
        path,
        K_max=K_max,
        sparse_factor=int(args.sparse_factor),
        seed=int(args.seed),
    )
    print("[dataset] len =", len(ds), "K_max =", K_max)

    rng = random.Random(int(args.seed))
    n = min(int(args.num), len(ds))
    idxs = [rng.randrange(0, len(ds)) for _ in range(n)]

    bad = 0
    first_bad = None
    for t, i in enumerate(idxs):
        errs = check_one(ds, i)
        if errs:
            bad += 1
            if first_bad is None:
                first_bad = i
            print("\n=== BAD idx =", i, " (#", bad, "at step", t, ") ===")
            for e in errs:
                print("  ", e)

            # stop early if lots of failures
            if bad >= 20:
                print("\n[STOP] too many bad samples, stop early.")
                break

    print("\n[summary] checked =", n, "bad =", bad, "first_bad =", first_bad)


if __name__ == "__main__":
    main()