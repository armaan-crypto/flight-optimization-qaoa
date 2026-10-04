import math
import sqlite3
import heapq
import time

from itertools import count
from collections import defaultdict
from datetime import datetime

from fare_evolution import project_fare


# ============================================================
# CONFIG — HARD CONSTRAINTS
# Flights violating these are dropped before scoring.
# ============================================================

SHOW_TOP_10_PATHS = True

MIN_LAYOVER_HOURS       = 1      # hard floor on layover
MAX_LAYOVER_HOURS       = 6      # hard ceiling on layover
MAX_SEGMENTS            = 4      # hard cap on number of flight legs
MAX_TOTAL_TRAVEL_HOURS  = None   # hard cap on total elapsed trip time
                                 # (None = no cap; e.g. 18 = reject > 18h)

DEFAULT_SEATS_REMAINING = 55

TIME_FMT = "%I:%M %p"


# ============================================================
# CONFIG — USER PREFERENCES (defaults, all passable per call)
# ============================================================

DEFAULT_USER_DESIRED_LAYOVER_HOURS = 2   # ideal layover length
IDEAL_LAYOVER_HOURS = DEFAULT_USER_DESIRED_LAYOVER_HOURS  # reference point for scoring

# Departure / arrival time windows (24-hour ints, inclusive).
# (0, 23) = no restriction.
DEFAULT_DEP_WINDOW_START = 0    # e.g. 6  = no flights before 6 AM
DEFAULT_DEP_WINDOW_END   = 23   # e.g. 21 = no flights after  9 PM
DEFAULT_ARR_WINDOW_START = 0
DEFAULT_ARR_WINDOW_END   = 23   # e.g. 22 = must land by 10 PM

# Preferred airlines get a scoring bonus (soft preference).
# Excluded airlines are hard-filtered out entirely.
DEFAULT_PREFERRED_AIRLINES = []   # e.g. ["United", "Delta"]
DEFAULT_EXCLUDED_AIRLINES  = []   # e.g. ["Spirit", "Frontier"]

# Hard filter — never route through these airports.
DEFAULT_AVOIDED_AIRPORTS = []     # e.g. ["ORD", "LGA"]


# ============================================================
# CONFIG — SCORING WEIGHTS
#
# Relative importance of each component. Do not need to sum
# to any value — they are normalized internally so the final
# weighted sum always lives in [0, 1] before sigmoid.
#
# Normalization ranges are derived from actual data at runtime
# (see NormStats), so no manual ceiling tuning is needed.
# ============================================================

WEIGHT_COST     = 3.0   # predicted fare            — dominant
WEIGHT_LAYOVER  = 1.0   # deviation from ideal layover
WEIGHT_SEGMENTS = 0.8   # prefer fewer stops
WEIGHT_AIRLINE  = 0.6   # preferred carrier bonus
WEIGHT_DEP_TIME = 0.3   # prefer mid-window departures
WEIGHT_TRAVEL   = 0.3   # prefer shorter flight legs

# Sigmoid shift. The weighted sum is shifted by this value before
# sigmoid so that a perfectly average route scores ~0.5.
# Raise toward 1.0 to push all scores lower (spread toward 0).
# Lower toward 0.0 to push all scores higher (spread toward 1).
SIGMOID_CENTER = 0.5


# ============================================================
# TIME PARSING
# ============================================================

def parse_time(raw):
    return datetime.strptime(raw.split(" on")[0], TIME_FMT)


# ============================================================
# FLIGHT OBJECT
# ============================================================

class Flight:

    __slots__ = (
        "id",
        "airline",
        "departure_airport",
        "arrival_airport",
        "departure_time",
        "arrival_time",
        "duration_minutes",
        "snapshot_price",
        "cabin",
        "is_best_flight",
    )

    def __init__(self, row):

        (
            self.id,
            self.airline,
            _flight_date,
            self.departure_airport,
            self.arrival_airport,
            dep_raw,
            arr_raw,
            self.duration_minutes,
            self.snapshot_price,
            self.cabin,
            self.is_best_flight,
        ) = row

        self.departure_time = parse_time(dep_raw)
        self.arrival_time   = parse_time(arr_raw)

    def __repr__(self):
        return (
            f"{self.departure_airport}->{self.arrival_airport}"
            f" ({self.airline})"
        )


# ============================================================
# LOAD FLIGHTS
# ============================================================

def load_flights(db_path="nonstop.db"):

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("""
        SELECT
            id,
            segmentsAirlineName,
            flightDate,
            segmentsDepartureAirportCode,
            segmentsArrivalAirportCode,
            segmentsDepartureTimeRaw,
            segmentsArrivalTimeRaw,
            travelDuration,
            totalFare,
            segmentsCabinCode,
            isBestFlight
        FROM flights
    """)

    flights = [Flight(row) for row in cur.fetchall()]

    conn.close()

    return flights


# ============================================================
# FARE CACHE
# ============================================================

fare_cache = {}


def get_predicted_fare(flight):

    if flight.id in fare_cache:
        return fare_cache[flight.id]

    predicted = project_fare(
        snapshot_price=flight.snapshot_price,
        snapshot_days_out=20,
        target_days_out=19,
        airline=flight.airline,
        cabin=flight.cabin,
        is_nonstop=True,
        seats_remaining=DEFAULT_SEATS_REMAINING,
    )

    fare_cache[flight.id] = predicted

    return predicted


# ============================================================
# GRAPH BUILDING
# ============================================================

def build_graph(flights):

    adjacency = defaultdict(list)

    for flight in flights:
        adjacency[flight.departure_airport].append(flight)

    for airport in adjacency:
        adjacency[airport].sort(key=lambda f: f.departure_time)

    return adjacency


# ============================================================
# NORMALIZATION STATS  (computed once from data, not hardcoded)
# ============================================================

class NormStats:
    """
    Reference-point normalization parameters.

    Each component has an *ideal* value (penalty = 0.0) and a
    *worst-case deviation* (penalty = 1.0). Everything in between is
    proportional, making weights directly comparable across components.
    """
    __slots__ = (
        "fare_ideal", "fare_worst",
        "dur_ideal",  "dur_worst",
        "lay_ideal",  "lay_worst",
        "seg_worst",
        "dep_midpoint", "dep_half_width",
        "w_sum",
    )

    def __init__(self, fare_ideal, fare_worst, dur_ideal, dur_worst,
                 lay_ideal, lay_worst, seg_worst,
                 dep_midpoint, dep_half_width, w_sum):
        self.fare_ideal     = fare_ideal
        self.fare_worst     = fare_worst
        self.dur_ideal      = dur_ideal
        self.dur_worst      = dur_worst
        self.lay_ideal      = lay_ideal
        self.lay_worst      = lay_worst
        self.seg_worst      = seg_worst
        self.dep_midpoint   = dep_midpoint
        self.dep_half_width = dep_half_width
        self.w_sum          = w_sum


def compute_norm_stats(flights, desired_layover_hours,
                       dep_window_start=DEFAULT_DEP_WINDOW_START,
                       dep_window_end=DEFAULT_DEP_WINDOW_END):
    """
    Single pass over all flights to establish reference points.

    Cost and duration ideals are data-driven (actual cheapest/fastest).
    Layover, segment, and departure-time ideals come from config/prefs.
    """
    fares = [get_predicted_fare(f) for f in flights]
    durs  = [f.duration_minutes    for f in flights]

    fare_min, fare_max = min(fares), max(fares)
    dur_min,  dur_max  = min(durs),  max(durs)

    # Cost: ideal = cheapest flight; worst deviation = full observed range
    fare_ideal = fare_min
    fare_worst = max(fare_max - fare_min, 1.0)

    # Duration: ideal = shortest leg; worst deviation = full observed range
    dur_ideal = dur_min
    dur_worst = max(dur_max - dur_min, 1.0)

    # Layover: ideal = desired hours; worst = farthest within hard window
    lay_ideal = desired_layover_hours
    lay_worst = max(
        MAX_LAYOVER_HOURS - desired_layover_hours,
        desired_layover_hours - MIN_LAYOVER_HOURS,
        0.01,
    )

    # Segments: ideal = direct (0 prior legs); worst = MAX_SEGMENTS - 1
    seg_worst = float(max(MAX_SEGMENTS - 1, 1))

    # Departure time: ideal = midpoint of window; worst = window edge
    dep_midpoint   = (dep_window_start + dep_window_end) / 2.0
    dep_half_width = max((dep_window_end - dep_window_start) / 2.0, 0.01)

    w_sum = (WEIGHT_COST + WEIGHT_LAYOVER + WEIGHT_SEGMENTS
             + WEIGHT_AIRLINE + WEIGHT_DEP_TIME + WEIGHT_TRAVEL)

    return NormStats(fare_ideal, fare_worst, dur_ideal, dur_worst,
                     lay_ideal, lay_worst, seg_worst,
                     dep_midpoint, dep_half_width, w_sum)


def _refpoint(value, ideal, worst_deviation):
    """
    Reference-point penalty: 0.0 when value == ideal, 1.0 at
    worst_deviation away from ideal.  Clamped to [0, 1].
    Works symmetrically for both directions from ideal.
    """
    return max(0.0, min(1.0, abs(value - ideal) / worst_deviation))


def _sigmoid(x):
    """Logistic function → (0, 1).  Numerically stable for large |x|."""
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================
# HEURISTIC
# ============================================================

def build_heuristic(
    flights,
    destination,
    ns,
    preferred_airlines=DEFAULT_PREFERRED_AIRLINES,
):
    """
    Admissible lower-bound on remaining sigmoid-score from each airport
    to the destination.

    Best-case assumptions (never overestimate):
      COST     : minimum observed fare to destination (data-driven)
      LAYOVER  : 0.0  (could land exactly at desired layover)
      SEGMENTS : 0.0  (no extra stops charged at this step)
      AIRLINE  : 0.0  (assume preferred carrier is available)
      DEP_TIME : 0.0  (assume perfect mid-window departure)
      TRAVEL   : minimum observed duration to destination (data-driven)
    """
    best_fare = {}
    best_dur  = {}

    for f in flights:
        if f.arrival_airport != destination:
            continue
        fare = get_predicted_fare(f)
        dur  = f.duration_minutes
        if f.departure_airport not in best_fare or fare < best_fare[f.departure_airport]:
            best_fare[f.departure_airport] = fare
        if f.departure_airport not in best_dur  or dur  < best_dur[f.departure_airport]:
            best_dur[f.departure_airport]  = dur

    heuristic = defaultdict(lambda: 0.0)

    for airport in best_fare:
        # Reference-point penalties — best case only (admissibility)
        pen_cost   = _refpoint(best_fare[airport], ns.fare_ideal, ns.fare_worst)
        pen_travel = _refpoint(best_dur[airport],  ns.dur_ideal,  ns.dur_worst)

        weighted_sum = (
            WEIGHT_COST     * pen_cost
            + WEIGHT_LAYOVER  * 0.0
            + WEIGHT_SEGMENTS * 0.0
            + WEIGHT_AIRLINE  * 0.0
            + WEIGHT_DEP_TIME * 0.0
            + WEIGHT_TRAVEL   * pen_travel
        ) / ns.w_sum

        heuristic[airport] = _sigmoid(weighted_sum - SIGMOID_CENTER)

    return heuristic


# ============================================================
# EDGE COST
# ============================================================

def edge_cost(
    flight,
    layover_minutes,
    segments_used,
    ns,
    desired_layover_hours=DEFAULT_USER_DESIRED_LAYOVER_HOURS,
    preferred_airlines=DEFAULT_PREFERRED_AIRLINES,
    dep_window_start=DEFAULT_DEP_WINDOW_START,
    dep_window_end=DEFAULT_DEP_WINDOW_END,
):
    """
    Normalized, weighted, sigmoid score for one flight leg.

    Pipeline
    --------
      1. Compute raw value for each component
      2. Min-max normalize each to [0, 1] using NormStats
      3. Multiply by component weight
      4. Sum and divide by w_sum  →  weighted_sum ∈ [0, 1]
      5. Shift by SIGMOID_CENTER, apply sigmoid  →  score ∈ (0, 1)

    Lower score = better.

    Returns (score, breakdown) where breakdown is a dict with every
    normalized component value, for display in the top-10 output.
    """

    fare = get_predicted_fare(flight)

    # ── 1. Reference-point penalty per component → [0, 1] ────────────────
    #       0.0 = at ideal value,  1.0 = worst plausible deviation

    # COST: ideal = cheapest observed fare in dataset
    pen_cost = _refpoint(fare, ns.fare_ideal, ns.fare_worst)

    # LAYOVER: ideal = desired hours; 0 penalty on first leg (no layover)
    if segments_used > 0:
        actual_lay_hours = layover_minutes / 60.0
        pen_layover = _refpoint(actual_lay_hours, ns.lay_ideal, ns.lay_worst)
    else:
        actual_lay_hours = 0.0
        pen_layover = 0.0

    # SEGMENTS: ideal = direct flight (0 prior legs = 0 penalty)
    pen_segments = _refpoint(float(segments_used), 0.0, ns.seg_worst)

    # AIRLINE: binary — 0.0 preferred, 1.0 not preferred (0.0 if no pref set)
    if preferred_airlines:
        pen_airline = 0.0 if flight.airline in preferred_airlines else 1.0
    else:
        pen_airline = 0.0

    # DEP_TIME: ideal = midpoint of window; penalty grows toward window edges
    pen_dep = _refpoint(float(flight.departure_time.hour),
                        ns.dep_midpoint, ns.dep_half_width)

    # TRAVEL: ideal = shortest observed leg in dataset
    pen_travel = _refpoint(flight.duration_minutes, ns.dur_ideal, ns.dur_worst)

    # ── 2. Weighted sum divided by w_sum → [0, 1] ────────────────────────

    weighted_sum = (
        WEIGHT_COST     * pen_cost
        + WEIGHT_LAYOVER  * pen_layover
        + WEIGHT_SEGMENTS * pen_segments
        + WEIGHT_AIRLINE  * pen_airline
        + WEIGHT_DEP_TIME * pen_dep
        + WEIGHT_TRAVEL   * pen_travel
    ) / ns.w_sum

    # ── 3. Sigmoid ────────────────────────────────────────────────────────

    score = _sigmoid(weighted_sum - SIGMOID_CENTER)

    breakdown = {
        "pen_cost":     pen_cost,
        "pen_layover":  pen_layover,
        "pen_segments": pen_segments,
        "pen_airline":  pen_airline,
        "pen_dep":      pen_dep,
        "pen_travel":   pen_travel,
        "weighted_sum": weighted_sum,
        "score":        score,
    }

    return score, breakdown


# ============================================================
# PATH PRINTING
# ============================================================

def format_minutes(minutes):
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h}h {m:02d}m"


def print_path(path):

    if not path:
        return

    route = [path[0].departure_airport]
    for flight in path:
        route.append(flight.arrival_airport)
    print(" -> ".join(route))

    total_layover_minutes = 0.0
    total_flight_minutes  = 0.0

    for i, flight in enumerate(path):

        dep_str = flight.departure_time.strftime("%I:%M %p").lstrip("0")

        print(
            f"  {flight.departure_airport}"
            f" -> {flight.arrival_airport}"
            f" | departs {dep_str}"
            f" | {flight.airline}"
            f" | ${get_predicted_fare(flight):.2f}"
            f" | flight time: {format_minutes(flight.duration_minutes)}"
        )

        total_flight_minutes += flight.duration_minutes

        if i > 0:
            layover_minutes = (
                flight.departure_time - path[i - 1].arrival_time
            ).total_seconds() / 60
            total_layover_minutes += layover_minutes
            print(f"    layover at {flight.departure_airport}: {format_minutes(layover_minutes)}")

    if len(path) > 1:
        print(f"  total layover:     {format_minutes(total_layover_minutes)}")

    total_travel_minutes = total_flight_minutes + total_layover_minutes
    print(f"  total travel time: {format_minutes(total_travel_minutes)}")


# ============================================================
# A*
# ============================================================

def astar(
    adjacency,
    flights,
    origin,
    destination,
    desired_layover_hours=DEFAULT_USER_DESIRED_LAYOVER_HOURS,
    dep_window_start=DEFAULT_DEP_WINDOW_START,
    dep_window_end=DEFAULT_DEP_WINDOW_END,
    arr_window_start=DEFAULT_ARR_WINDOW_START,
    arr_window_end=DEFAULT_ARR_WINDOW_END,
    preferred_airlines=DEFAULT_PREFERRED_AIRLINES,
    excluded_airlines=DEFAULT_EXCLUDED_AIRLINES,
    avoided_airports=DEFAULT_AVOIDED_AIRPORTS,
    max_total_travel_hours=MAX_TOTAL_TRAVEL_HOURS,
):
    # Compute reference-point normalization stats from data (single pass)
    ns = compute_norm_stats(
        flights,
        desired_layover_hours,
        dep_window_start=dep_window_start,
        dep_window_end=dep_window_end,
    )

    heuristic = build_heuristic(
        flights,
        destination,
        ns=ns,
        preferred_airlines=preferred_airlines,
    )

    max_total_travel_minutes = (
        max_total_travel_hours * 60 if max_total_travel_hours is not None else None
    )

    start_runtime = time.perf_counter()

    pq      = []
    counter = count()

    # heap entry:
    # (f_score, g_score, raw_fare_total, unique_counter,
    #  airport, arrival_time, segments_used, elapsed_minutes, path)
    heapq.heappush(
        pq,
        (
            heuristic[origin],
            0.0,
            0.0,
            next(counter),
            origin,
            None,
            0,
            0.0,
            [],
        ),
    )

    best_seen       = {}
    completed_paths = []
    nodes_expanded  = 0

    while pq:

        (
            f_score,
            g_score,
            raw_fare_total,
            _,
            airport,
            current_arrival_time,
            segments_used,
            elapsed_minutes,
            path,
        ) = heapq.heappop(pq)

        nodes_expanded += 1

        if airport == destination and segments_used > 0:

            completed_paths.append((g_score, raw_fare_total, path))

            if not SHOW_TOP_10_PATHS:
                runtime = time.perf_counter() - start_runtime
                return {
                    "best_score":     g_score,
                    "best_fare":      raw_fare_total,
                    "best_path":      path,
                    "nodes_expanded": nodes_expanded,
                    "runtime":        runtime,
                }

            if len(completed_paths) >= 10:
                break

        if segments_used >= MAX_SEGMENTS:
            continue

        state_key = (airport, segments_used)
        if state_key in best_seen and best_seen[state_key] <= g_score:
            continue
        best_seen[state_key] = g_score

        for flight in adjacency.get(airport, []):

            # ── Hard filter: excluded airlines ───────────────────────────
            if excluded_airlines and flight.airline in excluded_airlines:
                continue

            # ── Hard filter: avoided connecting airports ──────────────────
            if (
                avoided_airports
                and flight.arrival_airport in avoided_airports
                and flight.arrival_airport != destination
            ):
                continue

            # ── Hard filter: departure time window ────────────────────────
            if not (dep_window_start <= flight.departure_time.hour <= dep_window_end):
                continue

            # ── Hard filter: arrival time window ──────────────────────────
            if not (arr_window_start <= flight.arrival_time.hour <= arr_window_end):
                continue

            # ── Hard filter: layover bounds ───────────────────────────────
            if current_arrival_time is not None:
                layover_minutes = (
                    flight.departure_time - current_arrival_time
                ).total_seconds() / 60
                if layover_minutes < MIN_LAYOVER_HOURS * 60:
                    continue
                if layover_minutes > MAX_LAYOVER_HOURS * 60:
                    continue
            else:
                layover_minutes = 0.0

            # ── Hard filter: max total travel time ────────────────────────
            new_elapsed = elapsed_minutes + layover_minutes + flight.duration_minutes
            if max_total_travel_minutes is not None:
                if new_elapsed > max_total_travel_minutes:
                    continue

            # ── Normalized, weighted, sigmoid edge score ─────────────────
            leg_score, _ = edge_cost(
                flight=flight,
                layover_minutes=layover_minutes,
                segments_used=segments_used,
                ns=ns,
                desired_layover_hours=desired_layover_hours,
                preferred_airlines=preferred_airlines,
                dep_window_start=dep_window_start,
                dep_window_end=dep_window_end,
            )
            new_g = g_score + leg_score

            new_raw_fare_total = raw_fare_total + get_predicted_fare(flight)
            new_path           = path + [flight]
            h                  = heuristic[flight.arrival_airport]
            new_f              = new_g + h

            heapq.heappush(
                pq,
                (
                    new_f,
                    new_g,
                    new_raw_fare_total,
                    next(counter),
                    flight.arrival_airport,
                    flight.arrival_time,
                    segments_used + 1,
                    new_elapsed,
                    new_path,
                ),
            )

    runtime = time.perf_counter() - start_runtime

    if not completed_paths:
        return {
            "best_score":     None,
            "best_fare":      None,
            "best_path":      [],
            "nodes_expanded": nodes_expanded,
            "runtime":        runtime,
        }

    completed_paths.sort(key=lambda x: x[0])
    best_score, best_fare, best_path = completed_paths[0]

    if SHOW_TOP_10_PATHS:

        print("\n==============================")
        print("TOP 10 PATHS")
        print("==============================\n")

        for rank, (score, fare_total, path) in enumerate(completed_paths[:10], start=1):

            print(f"{rank}. SCORE = {score:.4f} | TOTAL FARE = ${fare_total:.2f}")

            # Per-segment score breakdown
            prev_arrival = None
            for i, flight in enumerate(path):
                layover_min = 0.0
                if prev_arrival is not None:
                    layover_min = (
                        flight.departure_time - prev_arrival
                    ).total_seconds() / 60
                _, bd = edge_cost(
                    flight=flight,
                    layover_minutes=layover_min,
                    segments_used=i,
                    ns=ns,
                    desired_layover_hours=desired_layover_hours,
                    preferred_airlines=preferred_airlines,
                    dep_window_start=dep_window_start,
                    dep_window_end=dep_window_end,
                )
                print(
                    f"  leg {i+1} {flight.departure_airport}->{flight.arrival_airport}"
                    f"  score={bd['score']:.4f}"
                    f"  [cost={bd['pen_cost']:.2f}"
                    f"  lay={bd['pen_layover']:.2f}"
                    f"  seg={bd['pen_segments']:.2f}"
                    f"  airline={bd['pen_airline']:.2f}"
                    f"  dep={bd['pen_dep']:.2f}"
                    f"  travel={bd['pen_travel']:.2f}]"
                )
                prev_arrival = flight.arrival_time

            print_path(path)
            print()

    return {
        "best_score":     best_score,
        "best_fare":      best_fare,
        "best_path":      best_path,
        "nodes_expanded": nodes_expanded,
        "runtime":        runtime,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    flights   = load_flights("nonstop.db")
    adjacency = build_graph(flights)

    result = astar(
        adjacency=adjacency,
        flights=flights,
        origin="FAI",
        destination="AMA",
        desired_layover_hours=DEFAULT_USER_DESIRED_LAYOVER_HOURS,
        dep_window_start=DEFAULT_DEP_WINDOW_START,
        dep_window_end=DEFAULT_DEP_WINDOW_END,
        arr_window_start=DEFAULT_ARR_WINDOW_START,
        arr_window_end=DEFAULT_ARR_WINDOW_END,
        preferred_airlines=DEFAULT_PREFERRED_AIRLINES,
        excluded_airlines=DEFAULT_EXCLUDED_AIRLINES,
        avoided_airports=DEFAULT_AVOIDED_AIRPORTS,
        max_total_travel_hours=MAX_TOTAL_TRAVEL_HOURS,
    )

    if result["best_path"]:

        print("\n==============================")
        print("BEST ROUTE")
        print("==============================\n")

        print_path(result["best_path"])

        print("\n==============================")
        print("STATS")
        print("==============================")

        print(f"Score:          {result['best_score']:.4f}  (sigmoid, lower = better)")
        print(f"Total Fare:     ${result['best_fare']:.2f}")
        print(f"Nodes Expanded: {result['nodes_expanded']}")
        print(f"Runtime:        {result['runtime']:.4f} seconds")

    else:
        print("\nNo feasible route found under the current constraints.")
        print(f"Nodes Expanded: {result['nodes_expanded']}")
        print(f"Runtime:        {result['runtime']:.4f} seconds")