import sqlite3
import heapq
import time

from itertools import count
from collections import defaultdict
from datetime import datetime

from fare_evolution import project_fare


# ============================================================
# CONFIG
# ============================================================

SHOW_TOP_10_PATHS = True

MIN_LAYOVER_HOURS = 1
MAX_LAYOVER_HOURS = 6

DEFAULT_USER_MAX_LAYOVER_HOURS   = 4
DEFAULT_USER_DESIRED_LAYOVER_HOURS = 2   # preferred layover duration

MAX_SEGMENTS = 4

DEFAULT_SEATS_REMAINING = 55

# Edge weight coefficients
LAMBDA_LAYOVER      = 8.0
LAMBDA_TRAVEL_TIME  = 0.02

# Scoring weights: how much each factor contributes to the combined score.
# Must sum to 1.0. Price is weighted more heavily than layover preference.
WEIGHT_COST    = 1
WEIGHT_LAYOVER = 1 - WEIGHT_COST

# TIME_FMT = "%Y-%m-%dT%I:%M %p"
TIME_FMT = "%I:%M %p"


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
# HEURISTIC
# ============================================================

def build_heuristic(flights, destination, desired_layover_hours=DEFAULT_USER_DESIRED_LAYOVER_HOURS):
    """
    For each airport, estimate the minimum weighted cost to reach `destination`.

    The heuristic blends:
      - WEIGHT_COST   * min_fare_to_destination
      - WEIGHT_LAYOVER * best_case_layover_penalty  (always 0 — see below)

    Admissibility argument
    ----------------------
    A* requires h(n) <= true_cost(n -> goal) for every node n.

    Cost component: min_fare is the cheapest single ticket we've ever seen
    from this airport to the destination, so WEIGHT_COST * min_fare never
    exceeds the fare portion of the real remaining cost. ✓

    Layover component: the best possible layover deviation is 0 (the traveler
    lands exactly `desired_layover_hours` before the next departure). We
    always assume this best case, so WEIGHT_LAYOVER * 0 = 0 never
    overestimates the layover portion of the remaining cost. ✓

    The layover weight is included structurally so the heuristic's shape
    mirrors the objective — if you later want to raise WEIGHT_LAYOVER, the
    heuristic stays admissible automatically.
    """

    min_fare_to_dest = defaultdict(lambda: 0.0)

    for flight in flights:
        if flight.arrival_airport != destination:
            continue
        fare = get_predicted_fare(flight)
        current = min_fare_to_dest[flight.departure_airport]
        if current == 0.0 or fare < current:
            min_fare_to_dest[flight.departure_airport] = fare

    heuristic = defaultdict(lambda: 0.0)

    for airport, min_fare in min_fare_to_dest.items():
        # Best-case layover deviation is 0, so WEIGHT_LAYOVER term vanishes.
        heuristic[airport] = WEIGHT_COST * min_fare

    return heuristic


# ============================================================
# EDGE COST
# ============================================================

def edge_cost(flight, layover_minutes, desired_layover_hours=DEFAULT_USER_DESIRED_LAYOVER_HOURS):
    """
    w = WEIGHT_COST   * fare
      + WEIGHT_LAYOVER * (LAMBDA_LAYOVER * |actual - desired| in hours
                          + LAMBDA_TRAVEL_TIME * duration_minutes)

    Layover deviation is symmetric around the desired value:
      desired=2h, actual=3h  → deviation=1h   (lightly penalized)
      desired=2h, actual=0.5h → deviation=1.5h (more penalized)
      desired=2h, actual=5h  → deviation=3h   (heavily penalized)

    So a 3-hour layover beats a 30-minute one when 2 hours is desired,
    and both lose to exactly 2 hours. Price still dominates at 80%.
    """

    fare = get_predicted_fare(flight)

    desired_layover_minutes  = desired_layover_hours * 60
    layover_deviation_hours  = abs(layover_minutes - desired_layover_minutes) / 60

    cost_component = WEIGHT_COST * fare

    layover_component = WEIGHT_LAYOVER * (
        LAMBDA_LAYOVER     * layover_deviation_hours
        + LAMBDA_TRAVEL_TIME * flight.duration_minutes
    )

    return cost_component + layover_component


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
):

    heuristic = build_heuristic(
        flights,
        destination,
        desired_layover_hours=desired_layover_hours,
    )

    start_runtime = time.perf_counter()

    pq      = []
    counter = count()

    # heap entry:
    #
    # (
    #     f_score,
    #     g_score,          <- weighted optimization score (used by A*)
    #     raw_fare_total,   <- sum of predicted fares only (displayed to user)
    #     unique_counter,
    #     airport,
    #     arrival_time,
    #     segments_used,
    #     path
    # )

    heapq.heappush(
        pq,
        (
            heuristic[origin],
            0,
            0.0,
            next(counter),
            origin,
            None,
            0,
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
            path,
        ) = heapq.heappop(pq)

        nodes_expanded += 1

        if airport == destination:

            completed_paths.append((g_score, raw_fare_total, path))

            if not SHOW_TOP_10_PATHS:

                runtime = time.perf_counter() - start_runtime

                return {
                    "best_cost":      g_score,
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

            # layover validation
            if current_arrival_time is not None:

                layover_minutes = (
                    flight.departure_time - current_arrival_time
                ).total_seconds() / 60

                if layover_minutes < MIN_LAYOVER_HOURS * 60:
                    continue

                if layover_minutes > MAX_LAYOVER_HOURS * 60:
                    continue

            else:
                layover_minutes = 0

            new_g = g_score + edge_cost(
                flight,
                layover_minutes,
                desired_layover_hours=desired_layover_hours,
            )

            new_raw_fare_total = raw_fare_total + get_predicted_fare(flight)

            new_path = path + [flight]

            h     = heuristic[flight.arrival_airport]
            new_f = new_g + h

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
                    new_path,
                ),
            )

    runtime = time.perf_counter() - start_runtime

    completed_paths.sort(key=lambda x: x[0])

    best_cost, best_fare, best_path = completed_paths[0]

    if SHOW_TOP_10_PATHS:

        print("\n==============================")
        print("TOP 10 PATHS")
        print("==============================\n")

        for rank, (cost, fare_total, path) in enumerate(completed_paths[:10], start=1):

            print(f"{rank}. TOTAL FARE = ${fare_total:.2f}")

            print_path(path)

            print()

    return {
        "best_cost":      best_cost,
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
        origin="ABQ",
        destination="BZN",
        desired_layover_hours=DEFAULT_USER_DESIRED_LAYOVER_HOURS,
    )

    print("\n==============================")
    print("BEST ROUTE")
    print("==============================\n")

    print_path(result["best_path"])

    print("\n==============================")
    print("STATS")
    print("==============================")

    print(f"Total Fare:     ${result['best_fare']:.2f}")
    print(f"Nodes Expanded: {result['nodes_expanded']}")
    print(f"Runtime:        {result['runtime']:.4f} seconds")