"""
testing.py
==========
Stress-test benchmark: A* (astar2.py) vs QAOA (qaoa.py) across many routes
and network sizes, run in parallel using multiple CPU cores.

Timeout design
--------------
Each task runs in a dedicated child process (multiprocessing.Process).
The main process joins each child with TASK_TIMEOUT_SECONDS; if the child
is still alive after that, it is hard-killed with terminate() + kill().
This is the only reliable way to stop a hanging solver — thread timeouts
and ProcessPoolExecutor.result(timeout=…) only cancel the *wait*, not the
worker itself.

Parallelism design
------------------
A fixed pool of N_WORKERS slots is maintained manually.  When a slot frees
up (child exits or is killed) the next task is dispatched.  Results are
returned via a multiprocessing.Queue so the child never writes to the CSV
directly — the main process owns all I/O.

Route coverage
--------------
  FIXED_ROUTES   — hand-picked pairs covering all haul lengths and topologies
  STRESS_ROUTES  — edge cases: remote airports, reversed pairs, O=D
  RANDOM_ROUTES  — N_RANDOM_ROUTES pairs sampled from DB airports (seeded)

network_size="full" is stored as -1 in the CSV and labelled "full" only in
display output, so analyze.py can treat every network_size as an integer.

Execution
---------
  python testing.py                   # uses os.cpu_count() workers
  python testing.py --workers 4       # cap at 4
  python testing.py --timeout 60      # per-task timeout in seconds
  python testing.py --dry-run         # print plan, don't run
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH             = "nonstop.db"
RESULTS_CSV         = "results.csv"
RESUME              = True

TRIALS_PER_CFG      = 5      # repetitions per (origin, dest, network_size)
N_RANDOM_ROUTES     = 30     # random pairs drawn from DB airports
RANDOM_SEED         = 42
TASK_TIMEOUT_SEC    = 90     # hard kill after this many seconds per task
N_WORKERS           = None   # None → os.cpu_count()

# Network sizes. Use -1 as the sentinel for "full graph" (no subsampling).
# Stored as -1 in CSV; analyze.py maps -1 → "full" for display.
NETWORK_SIZES = [50, 100, 200, 300, 500, -1]

FIXED_ROUTES = [
    # short-haul
    ("DEN", "MSP"), ("LAX", "SFO"), ("BOS", "DCA"), ("DAL", "HOU"),
    # medium-haul
    ("ABQ", "BZN"), ("SEA", "PHX"), ("ORD", "ATL"), ("SLC", "DFW"),
    # long-haul
    ("LAX", "BOS"), ("SEA", "MIA"), ("ANC", "MIA"), ("FAI", "AMA"),
    # hub-to-hub
    ("ATL", "LAX"), ("ORD", "DFW"), ("JFK", "LAX"),
    # hub-to-regional
    ("ATL", "BZN"), ("LAX", "FAR"), ("DFW", "FCA"),
    # regional-to-regional (hardest)
    ("BZN", "MSO"), ("FAI", "JNU"), ("GCC", "CPR"), ("JAC", "RAP"),
]

STRESS_ROUTES = [
    ("BRD", "HNL"),   # tiny Minnesota → Hawaii
    ("DIK", "SJU"),   # Dickinson ND → Puerto Rico
    ("TWF", "MYR"),   # Twin Falls → Myrtle Beach
    ("BZN", "ABQ"),   # reversed fixed route
    ("MIA", "ANC"),   # reversed long-haul
    ("AMA", "FAI"),   # reversed remote
    ("ATL", "ATL"),   # O = D (should return no route)
    ("FAI", "BOS"),
    ("BOS", "MIA"),
    ("MIA", "LAX"),
]

# Constraint pass-through — keep in sync with astar2.py / qaoa.py
MIN_LAYOVER_HOURS     = 1
MAX_LAYOVER_HOURS     = 6
MAX_SEGMENTS          = 4
DEP_WINDOW_START      = 0
DEP_WINDOW_END        = 23
ARR_WINDOW_START      = 0
ARR_WINDOW_END        = 23
DESIRED_LAYOVER_HOURS = 2.0

FULL_GRAPH_SENTINEL = -1   # stored in CSV when no subsampling

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "timestamp", "network_size", "origin", "destination", "trial_id",
    "solver", "total_route_cost", "total_fare_usd", "feasibility",
    "runtime_seconds", "nodes_expanded", "iterations", "num_legs",
    "optimality_gap", "route_string", "route_category", "timed_out",
]


def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_rows_csv(path, rows):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            w.writerow(row)


def load_completed(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((
                    row["origin"],
                    row["destination"],
                    int(row["network_size"]),
                    int(row["trial_id"]),
                    row["solver"],
                ))
            except (KeyError, ValueError):
                pass
    return done


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE POOL
# ══════════════════════════════════════════════════════════════════════════════

def build_route_pool(all_flights):
    airports = sorted(
        {f.departure_airport for f in all_flights}
        | {f.arrival_airport   for f in all_flights}
    )
    rng = random.Random(RANDOM_SEED)
    random_pairs = set()
    attempts = 0
    while len(random_pairs) < N_RANDOM_ROUTES and attempts < N_RANDOM_ROUTES * 20:
        o, d = rng.sample(airports, 2)
        if o != d:
            random_pairs.add((o, d))
        attempts += 1

    pool = []
    seen = set()

    def add(o, d, cat):
        if (o, d) not in seen:
            seen.add((o, d))
            pool.append((o, d, cat))

    for o, d in FIXED_ROUTES:
        add(o, d, "fixed")
    for o, d in STRESS_ROUTES:
        add(o, d, "stress")
    for o, d in random_pairs:
        add(o, d, "random")
    return pool


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH SUBSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def subsample_graph(all_flights, origin, destination, net_size):
    """net_size == FULL_GRAPH_SENTINEL → return all flights."""
    if net_size == FULL_GRAPH_SENTINEL:
        return all_flights

    all_ap = set()
    for f in all_flights:
        all_ap.add(f.departure_airport)
        all_ap.add(f.arrival_airport)

    fixed   = {origin, destination}
    others  = list(all_ap - fixed)
    slots   = max(0, net_size - len(fixed))
    sampled = fixed | set(random.sample(others, min(slots, len(others))))
    return [f for f in all_flights
            if f.departure_airport in sampled and f.arrival_airport in sampled]


# ══════════════════════════════════════════════════════════════════════════════
#  FEASIBILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_feasibility(path, origin, destination):
    if not path:
        return 0
    if path[0].departure_airport != origin:
        return 0
    if path[-1].arrival_airport != destination:
        return 0
    if len(path) > MAX_SEGMENTS:
        return 0
    visited = {origin}
    for i, flight in enumerate(path):
        if not (DEP_WINDOW_START <= flight.departure_time.hour <= DEP_WINDOW_END):
            return 0
        if not (ARR_WINDOW_START <= flight.arrival_time.hour <= ARR_WINDOW_END):
            return 0
        arr = flight.arrival_airport
        if arr in visited and arr != destination:
            return 0
        visited.add(arr)
        if i > 0:
            lay = (flight.departure_time - path[i-1].arrival_time).total_seconds() / 60
            if lay < MIN_LAYOVER_HOURS * 60 or lay > MAX_LAYOVER_HOURS * 60:
                return 0
    return 1


# ══════════════════════════════════════════════════════════════════════════════
#  CANONICAL SCORER
# ══════════════════════════════════════════════════════════════════════════════

def score_path(path, ns, edge_cost_fn, get_fare_fn):
    total_cost = 0.0
    total_fare = 0.0
    for i, flight in enumerate(path):
        lay = 0.0
        if i > 0:
            lay = (flight.departure_time - path[i-1].arrival_time).total_seconds() / 60
        leg_score, _ = edge_cost_fn(
            flight=flight,
            layover_minutes=lay,
            segments_used=i,
            ns=ns,
            desired_layover_hours=DESIRED_LAYOVER_HOURS,
            preferred_airlines=[],
            dep_window_start=DEP_WINDOW_START,
            dep_window_end=DEP_WINDOW_END,
        )
        total_cost += leg_score
        total_fare += get_fare_fn(flight)
    return total_cost, total_fare


def route_string(path):
    if not path:
        return ""
    return "->".join([path[0].departure_airport] + [f.arrival_airport for f in path])


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER — runs inside a dedicated child process
# ══════════════════════════════════════════════════════════════════════════════

def _worker(task, result_queue):
    """
    Child-process entry point.  Puts a list of result dicts onto result_queue,
    then exits.  The main process reads from the queue after join/kill.

    Imports happen here so each worker gets its own module state — critical
    for qaoa.py's mutable globals.
    """
    origin, destination, net_size, trial_id, category, seed = task

    import astar2
    from astar2 import (
        load_flights       as astar_load,
        build_graph        as astar_graph,
        compute_norm_stats as astar_norm,
        edge_cost          as astar_edge,
        get_predicted_fare as astar_fare,
        astar,
    )
    import qaoa as qaoa_mod
    from qaoa import (
        build_graph                as qaoa_graph,
        build_itinerary_candidates as qaoa_candidates,
        compute_norm_stats         as qaoa_norm,
    )

    try:
        all_flights = astar_load(DB_PATH)
    except Exception as exc:
        result_queue.put([])
        return

    random.seed(seed)
    flights = subsample_graph(all_flights, origin, destination, net_size)

    if not flights:
        result_queue.put([])
        return

    ts   = datetime.utcnow().isoformat(timespec="seconds")
    base = {
        "timestamp":      ts,
        "network_size":   net_size,          # integer; -1 for full graph
        "origin":         origin,
        "destination":    destination,
        "trial_id":       trial_id,
        "route_category": category,
        "timed_out":      0,
    }

    results    = []
    astar_cost = None

    # ── A* ────────────────────────────────────────────────────────────────────
    try:
        ns        = astar_norm(flights, DESIRED_LAYOVER_HOURS,
                               dep_window_start=DEP_WINDOW_START,
                               dep_window_end=DEP_WINDOW_END)
        adjacency = astar_graph(flights)
        t0        = time.perf_counter()
        result    = astar(
            adjacency=adjacency, flights=flights,
            origin=origin, destination=destination,
            desired_layover_hours=DESIRED_LAYOVER_HOURS,
            dep_window_start=DEP_WINDOW_START, dep_window_end=DEP_WINDOW_END,
            arr_window_start=ARR_WINDOW_START, arr_window_end=ARR_WINDOW_END,
            preferred_airlines=[], excluded_airlines=[], avoided_airports=[],
            max_total_travel_hours=None,
        )
        rt       = time.perf_counter() - t0
        path     = result.get("best_path", [])
        feasible = check_feasibility(path, origin, destination)
        cost, fare = score_path(path, ns, astar_edge, astar_fare) if path else (None, None)
        astar_cost = cost
        results.append({**base,
            "solver": "astar",
            "total_route_cost": round(cost, 6) if cost is not None else "",
            "total_fare_usd":   round(fare, 2) if fare is not None else "",
            "feasibility":      feasible,
            "runtime_seconds":  round(rt, 4),
            "nodes_expanded":   result.get("nodes_expanded", ""),
            "iterations":       "",
            "num_legs":         len(path),
            "optimality_gap":   0.0,
            "route_string":     route_string(path),
        })
    except Exception as exc:
        results.append({**base,
            "solver": "astar", "feasibility": 0, "runtime_seconds": 0,
            "optimality_gap": "", "route_string": f"ERROR:{exc}",
            "total_route_cost": "", "total_fare_usd": "",
            "nodes_expanded": "", "iterations": "", "num_legs": 0,
        })

    # ── QAOA ──────────────────────────────────────────────────────────────────
    try:
        qaoa_mod.ORIGIN      = origin
        qaoa_mod.DESTINATION = destination
        qaoa_mod._ns         = qaoa_norm(flights)
        adjacency_q          = qaoa_graph(flights)
        t0                   = time.perf_counter()
        candidates           = qaoa_candidates(adjacency_q)
        rt                   = time.perf_counter() - t0
        path                 = candidates[0] if candidates else []
        feasible             = check_feasibility(path, origin, destination)
        ns_score = astar_norm(flights, DESIRED_LAYOVER_HOURS,
                              dep_window_start=DEP_WINDOW_START,
                              dep_window_end=DEP_WINDOW_END)
        cost, fare = score_path(path, ns_score, astar_edge, astar_fare) if path else (None, None)
        gap = (round((cost - astar_cost) / astar_cost, 6)
               if isinstance(astar_cost, float) and isinstance(cost, float) and astar_cost > 0
               else "")
        results.append({**base,
            "solver": "qaoa",
            "total_route_cost": round(cost, 6) if cost is not None else "",
            "total_fare_usd":   round(fare, 2) if fare is not None else "",
            "feasibility":      feasible,
            "runtime_seconds":  round(rt, 4),
            "nodes_expanded":   "",
            "iterations":       len(candidates),
            "num_legs":         len(path),
            "optimality_gap":   gap,
            "route_string":     route_string(path),
        })
    except Exception as exc:
        results.append({**base,
            "solver": "qaoa", "feasibility": 0, "runtime_seconds": 0,
            "optimality_gap": "", "route_string": f"ERROR:{exc}",
            "total_route_cost": "", "total_fare_usd": "",
            "nodes_expanded": "", "iterations": "", "num_legs": 0,
        })

    result_queue.put(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK RUNNER WITH HARD TIMEOUT
# ══════════════════════════════════════════════════════════════════════════════

def run_task_with_timeout(task, timeout_sec):
    """
    Spawn a child process for `task`.  If it doesn't finish within timeout_sec,
    hard-kill it and return a timed-out sentinel row pair.
    Returns a list of result dicts (0, 1, or 2 entries).
    """
    origin, destination, net_size, trial_id, category, seed = task
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(task, q), daemon=True)
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        # Hard kill — terminate() sends SIGTERM, kill() sends SIGKILL
        p.terminate()
        p.join(2)
        if p.is_alive():
            p.kill()
            p.join(1)

        # Write timed-out sentinel rows so the CSV records the attempt
        ts   = datetime.utcnow().isoformat(timespec="seconds")
        base = {
            "timestamp": ts, "network_size": net_size,
            "origin": origin, "destination": destination,
            "trial_id": trial_id, "route_category": category, "timed_out": 1,
            "feasibility": 0, "runtime_seconds": timeout_sec,
            "total_route_cost": "", "total_fare_usd": "",
            "nodes_expanded": "", "iterations": "", "num_legs": 0,
            "optimality_gap": "", "route_string": "TIMEOUT",
        }
        return [
            {**base, "solver": "astar"},
            {**base, "solver": "qaoa"},
        ]

    # Process finished in time — drain the queue
    try:
        return q.get_nowait()
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER POOL (manual, so we can call terminate() on hanging processes)
# ══════════════════════════════════════════════════════════════════════════════

def run_pool(tasks, n_workers, timeout_sec, csv_path):
    """
    Simple fixed-size worker pool.  Each slot holds one (Process, Queue, task).
    When a slot's process exits or times out, the result is collected and the
    slot is refilled with the next task.

    We manage the pool manually rather than using ProcessPoolExecutor because
    Executor.submit() doesn't let us call p.terminate() on the underlying
    Process — it only cancels the Future's wait.
    """
    total      = len(tasks)
    done_count = 0
    errors     = 0
    task_iter  = iter(tasks)
    slots      = []   # list of (Process, Queue, task, start_time)

    def _dispatch(task):
        q = mp.Queue()
        p = mp.Process(target=_worker, args=(task, q), daemon=True)
        p.start()
        return p, q, task, time.monotonic()

    def _collect(p, q, task, start_time, timed_out=False):
        nonlocal done_count, errors
        done_count += 1
        origin, destination, net_size, trial_id, category, _ = task

        if timed_out:
            ts   = datetime.utcnow().isoformat(timespec="seconds")
            base = {
                "timestamp": ts, "network_size": net_size,
                "origin": origin, "destination": destination,
                "trial_id": trial_id, "route_category": category, "timed_out": 1,
                "feasibility": 0, "runtime_seconds": round(time.monotonic()-start_time, 2),
                "total_route_cost": "", "total_fare_usd": "",
                "nodes_expanded": "", "iterations": "", "num_legs": 0,
                "optimality_gap": "", "route_string": "TIMEOUT",
            }
            rows = [{**base, "solver": "astar"}, {**base, "solver": "qaoa"}]
            size_label = "full" if net_size == FULL_GRAPH_SENTINEL else net_size
            print(f"  [{done_count:>4}/{total}] TIMEOUT "
                  f"{origin}->{destination} size={size_label} trial={trial_id}")
            errors += 1
        else:
            try:
                rows = q.get_nowait()
            except Exception:
                rows = []
            for r in rows:
                solver = r.get("solver", "?")
                cost   = r.get("total_route_cost", "—")
                fare   = r.get("total_fare_usd", "—")
                rt     = r.get("runtime_seconds", "—")
                gap    = r.get("optimality_gap", "")
                gap_s  = f"  gap={gap}" if gap not in ("", 0.0) else ""
                size_label = "full" if net_size == FULL_GRAPH_SENTINEL else net_size
                print(f"  [{done_count:>4}/{total}] {solver:<5} "
                      f"{origin}->{destination:<4} size={str(size_label):<5} "
                      f"trial={trial_id} cat={category:<6} "
                      f"cost={cost} fare=${fare} t={rt}s{gap_s}")

        if rows:
            append_rows_csv(csv_path, rows)

    # Fill initial slots
    for task in task_iter:
        if len(slots) >= n_workers:
            break
        slots.append(_dispatch(task))

    # Main loop
    while slots:
        time.sleep(0.05)
        still_running = []
        for p, q, task, start_time in slots:
            elapsed = time.monotonic() - start_time

            if not p.is_alive():
                # Finished normally
                _collect(p, q, task, start_time, timed_out=False)
                # Dispatch next task into this slot
                try:
                    next_task = next(task_iter)
                    still_running.append(_dispatch(next_task))
                except StopIteration:
                    pass  # no more tasks; slot closes

            elif elapsed > timeout_sec:
                # Hard kill
                p.terminate()
                p.join(2)
                if p.is_alive():
                    p.kill()
                    p.join(1)
                _collect(p, q, task, start_time, timed_out=True)
                try:
                    next_task = next(task_iter)
                    still_running.append(_dispatch(next_task))
                except StopIteration:
                    pass

            else:
                still_running.append((p, q, task, start_time))

        slots = still_running

    return done_count, errors


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",  type=int,   default=None,
                        help="Parallel worker processes (default: CPU count)")
    parser.add_argument("--timeout",  type=float, default=TASK_TIMEOUT_SEC,
                        help=f"Per-task timeout in seconds (default: {TASK_TIMEOUT_SEC})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print task plan without running")
    args = parser.parse_args()

    print(f"Loading flight database from {DB_PATH} …")
    try:
        from astar2 import load_flights as _load
        all_flights = _load(DB_PATH)
    except Exception as exc:
        sys.exit(f"Failed to load DB: {exc}")
    print(f"  {len(all_flights)} flights loaded.")

    route_pool = build_route_pool(all_flights)
    n_fixed  = sum(1 for *_, c in route_pool if c == "fixed")
    n_stress = sum(1 for *_, c in route_pool if c == "stress")
    n_random = sum(1 for *_, c in route_pool if c == "random")
    print(f"  {len(route_pool)} routes  "
          f"({n_fixed} fixed / {n_stress} stress / {n_random} random)")
    total_planned = len(route_pool) * len(NETWORK_SIZES) * TRIALS_PER_CFG * 2
    print(f"  {len(NETWORK_SIZES)} sizes × {TRIALS_PER_CFG} trials × 2 solvers "
          f"= {total_planned} rows planned\n")

    init_csv(RESULTS_CSV)
    completed = load_completed(RESULTS_CSV) if RESUME else set()

    tasks = []
    for origin, destination, category in route_pool:
        for net_size in NETWORK_SIZES:
            for trial_id in range(1, TRIALS_PER_CFG + 1):
                done_a = (origin, destination, net_size, trial_id, "astar") in completed
                done_q = (origin, destination, net_size, trial_id, "qaoa")  in completed
                if RESUME and done_a and done_q:
                    continue
                seed = hash((origin, destination, net_size, trial_id)) & 0xFFFFFFFF
                tasks.append((origin, destination, net_size, trial_id, category, seed))

    print(f"  {len(tasks)} tasks to run "
          f"({'resuming — skipped already-done rows' if RESUME else 'fresh start'}).")

    if args.dry_run:
        print("\nDRY RUN — first 20 tasks:")
        for t in tasks[:20]:
            size_label = "full" if t[2] == FULL_GRAPH_SENTINEL else t[2]
            print(f"  {t[0]}->{t[1]}  size={size_label}  trial={t[3]}  cat={t[4]}")
        if len(tasks) > 20:
            print(f"  … and {len(tasks)-20} more.")
        return

    if not tasks:
        print("Nothing to do — all trials already complete.")
        _print_summary(RESULTS_CSV)
        return

    n_workers = args.workers or os.cpu_count() or 2
    n_workers = min(n_workers, len(tasks))
    timeout   = args.timeout
    print(f"\nRunning {len(tasks)} tasks | {n_workers} workers | "
          f"{timeout}s timeout per task\n")

    t_start = time.monotonic()
    done, errors = run_pool(tasks, n_workers, timeout, RESULTS_CSV)
    elapsed = time.monotonic() - t_start

    print(f"\n{'='*65}")
    print(f"Finished: {done} tasks in {elapsed:.1f}s  ({errors} timeouts/errors)")
    print(f"Results → {RESULTS_CSV}")
    _print_summary(RESULTS_CSV)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(path):
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def sf(v):
        try:    return float(v)
        except: return None

    def ms(lst):
        vals = [v for v in lst if v is not None]
        if not vals:
            return float("nan"), float("nan")
        mu = sum(vals) / len(vals)
        sd = (sum((x-mu)**2 for x in vals) / max(len(vals)-1, 1)) ** 0.5
        return mu, sd

    groups = defaultdict(list)
    for r in rows:
        cat = r.get("route_category", "fixed")
        try:
            size = int(r["network_size"])
        except (KeyError, ValueError):
            size = -1
        groups[(r["solver"], size, cat)].append(r)

    print("\n" + "=" * 90)
    print("SUMMARY  (mean ± std, grouped by solver / size / category)")
    print("=" * 90)
    print(f"{'solver':<6} {'size':<6} {'cat':<8} {'n':>4}"
          f" {'cost μ±σ':<22} {'feas':>6} {'rt_s μ':>8} {'gap μ':>8} {'timeouts':>8}")
    print("-" * 90)

    for key in sorted(groups, key=lambda k: (k[0], k[2], k[1])):
        solver, size, cat = key
        recs     = groups[key]
        costs    = [sf(r["total_route_cost"]) for r in recs]
        feas     = [sf(r["feasibility"])      for r in recs]
        rts      = [sf(r["runtime_seconds"])  for r in recs]
        gaps     = [sf(r["optimality_gap"])   for r in recs if sf(r.get("optimality_gap")) is not None]
        timeouts = sum(1 for r in recs if sf(r.get("timed_out")) == 1)
        mc, sc   = ms(costs)
        mf, _    = ms(feas)
        mr, _    = ms(rts)
        mg, _    = ms(gaps) if gaps else (float("nan"), 0)
        size_lbl = "full" if size == FULL_GRAPH_SENTINEL else size
        print(f"{solver:<6} {str(size_lbl):<6} {cat:<8} {len(recs):>4}"
              f" {mc:>8.4f}±{sc:<10.4f} {mf:>6.3f} {mr:>8.4f} {mg:>8.4f} {timeouts:>8}")

    print("=" * 90)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # safe on macOS + Linux
    main()