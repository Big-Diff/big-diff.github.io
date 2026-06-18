# eval_official_pyvrp_gap.py
import argparse
import time
from collections import OrderedDict

import numpy as np
import pyvrp
import pyvrp.stop
from tqdm import tqdm


def build_model(locs, demand, caps, fixed_costs, unit_costs, coord_scale):
    m = pyvrp.Model()

    # 兼容两种 PyVRP Model API：
    # 1) 有 add_location(): 先建 location，再 add_depot/add_client
    # 2) 没有 add_location(): 直接 add_depot(x=..., y=...) 和 add_client(x=..., y=...)
    if hasattr(m, "add_location"):
        loc_objs = [m.add_location(float(x), float(y)) for x, y in locs]
        depot = m.add_depot(location=loc_objs[0], name="Depot")

        clients = []
        for i in range(1, len(loc_objs)):
            cli = m.add_client(
                location=loc_objs[i],
                delivery=int(demand[i - 1]),
                name=f"C{i}",
            )
            clients.append(cli)

        edge_nodes = loc_objs

    else:
        depot = m.add_depot(
            x=float(locs[0, 0]),
            y=float(locs[0, 1]),
            name="Depot",
        )

        clients = []
        for i in range(1, len(locs)):
            cli = m.add_client(
                x=float(locs[i, 0]),
                y=float(locs[i, 1]),
                delivery=int(demand[i - 1]),
                name=f"C{i}",
            )
            clients.append(cli)

        edge_nodes = [depot] + clients

    # 合并相同车辆配置
    groups = OrderedDict()
    for cap, fixed, unit in zip(caps, fixed_costs, unit_costs):
        key = (int(cap), int(fixed), int(unit))
        groups[key] = groups.get(key, 0) + 1

    for t, ((cap, fixed, unit), num) in enumerate(groups.items()):
        m.add_vehicle_type(
            num_available=int(num),
            capacity=int(cap),
            fixed_cost=int(fixed),
            unit_distance_cost=int(unit),
            start_depot=depot,
            end_depot=depot,
            name=f"type_{t}",
        )

    coords = np.asarray(locs, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    dist = np.rint(dist * coord_scale).astype(np.int64)

    for i, frm in enumerate(edge_nodes):
        for j, to in enumerate(edge_nodes):
            if i != j:
                d = int(dist[i, j])
                m.add_edge(frm, to, distance=d, duration=d)

    return m


def pyvrp_raw_cost(sol):
    fixed = int(sol.fixed_vehicle_cost()) if hasattr(sol, "fixed_vehicle_cost") else 0
    dist = int(sol.distance_cost()) if hasattr(sol, "distance_cost") else int(sol.distance())
    dur = int(sol.duration_cost()) if hasattr(sol, "duration_cost") else 0
    return fixed + dist + dur


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_npz",
        type=str,
        required=True,
        help="Path to generated solved HFVRP test npz.",
    )

    parser.add_argument(
        "--budgets_ms",
        type=int,
        nargs="+",
        required=True,
        help="PyVRP time budgets in milliseconds, e.g. --budgets_ms 50 100 200 500 1000",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of instances to evaluate. Default: all.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data = np.load(args.data_npz, allow_pickle=True)

    locs = data["locs"]
    demands = data["demand_linehaul_raw"]
    caps = data["vehicle_capacity_raw"]
    unit_costs = data["vehicle_unit_distance_cost_raw"]

    if "vehicle_fixed_cost_raw" in data:
        fixed_costs = data["vehicle_fixed_cost_raw"]
    else:
        fixed_costs = np.zeros_like(caps)

    # solved 测试集里的参考 cost
    ref_costs = data["costs"]

    coord_scale = int(np.asarray(data["coord_scale"]).reshape(-1)[0])
    unit_cost_scale = int(np.asarray(data["unit_cost_scale"]).reshape(-1)[0])
    obj_scale = coord_scale * unit_cost_scale

    n = len(locs) if args.count is None else min(args.count, len(locs))

    print(f"data_npz: {args.data_npz}")
    print(f"instances: {n}")
    print(f"budgets_ms: {args.budgets_ms}")
    print(f"coord_scale={coord_scale}, unit_cost_scale={unit_cost_scale}, obj_scale={obj_scale}")
    print()

    for budget_ms in args.budgets_ms:
        pyvrp_costs = []
        gaps = []
        times = []
        feasible = 0

        for i in tqdm(range(n), desc=f"PyVRP {budget_ms}ms"):
            model = build_model(
                locs=locs[i],
                demand=demands[i],
                caps=caps[i],
                fixed_costs=fixed_costs[i],
                unit_costs=unit_costs[i],
                coord_scale=coord_scale,
            )

            t0 = time.perf_counter()

            res = model.solve(
                stop=pyvrp.stop.MaxRuntime(float(budget_ms) / 1000.0),
                seed=args.seed + i,
                display=False,
            )

            wall_ms = (time.perf_counter() - t0) * 1000.0

            sol = res.best
            raw_cost = pyvrp_raw_cost(sol)
            cost = raw_cost / obj_scale

            gap = (cost - ref_costs[i]) / ref_costs[i] * 100.0

            pyvrp_costs.append(cost)
            gaps.append(gap)
            times.append(wall_ms)

            if sol.is_feasible():
                feasible += 1

        pyvrp_costs = np.asarray(pyvrp_costs, dtype=np.float64)
        gaps = np.asarray(gaps, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)

        print()
        print(f"========== PyVRP budget = {budget_ms} ms ==========")
        print(f"feasible:      {feasible}/{n}")
        print(f"cost mean:     {pyvrp_costs.mean():.6f}")
        print(f"gap mean:      {gaps.mean():.4f}%")
        print(f"gap std:       {gaps.std():.4f}%")
        print(f"time mean ms:  {times.mean():.2f}")
        print(f"time p95 ms:   {np.percentile(times, 95):.2f}")
        print()


if __name__ == "__main__":
    main()