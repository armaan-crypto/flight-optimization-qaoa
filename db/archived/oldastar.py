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

DEFAULT_USER_MAX_LAYOVER_HOURS = 4

MAX_SEGMENTS = 4

DEFAULT_SEATS_REMAINING = 55

# edge weight coefficients
LAMBDA_LAYOVER = 8.0
LAMBDA_TRAVEL_TIME = 0.02

TIME_FMT = "%Y-%m-%dT%I:%M %p"


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
        self.arrival_time = parse_time(arr_raw)

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
    cur = conn.cursor()

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

def build_heuristic(flights, destination):

    heuristic = defaultdict(lambda: 0)

    for flight in flights:

        if flight.arrival_airport != destination:
            continue

        fare = get_predicted_fare(flight)

        current = heuristic[flight.departure_airport]

        if current == 0 or fare < current:
            heuristic[flight.departure_airport] = fare

    return heuristic


# ============================================================
# EDGE COST
# ============================================================

def edge_cost(flight, layover_minutes):

    fare = get_predicted_fare(flight)

    return (
        fare
        + LAMBDA_LAYOVER * (layover_minutes / 60)
        + LAMBDA_TRAVEL_TIME * flight.duration_minutes
    )


# ============================================================
# PATH PRINTING
# ============================================================

def print_path(path):

    if not path:
        return

    route = [path[0].departure_airport]

    for flight in path:
        route.append(flight.arrival_airport)

    print(" -> ".join(route))

    for flight in path:

        print(
            f"  {flight.departure_airport}"
            f" -> {flight.arrival_airport}"
            f" | {flight.airline}"
            f" | ${get_predicted_fare(flight):.2f}"
        )


# ============================================================
# A*
# ============================================================

def astar(adjacency, flights, origin, destination):

    heuristic = build_heuristic(flights, destination)

    start_runtime = time.perf_counter()

    pq = []

    counter = count()

    # heap entry:
    #
    # (
    #     f_score,
    #     g_score,
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
            next(counter),
            origin,
            None,
            0,
            [],
        )
    )

    best_seen = {}

    completed_paths = []

    nodes_expanded = 0

    while pq:

        (
            f_score,
            g_score,
            _,
            airport,
            current_arrival_time,
            segments_used,
            path,
        ) = heapq.heappop(pq)

        nodes_expanded += 1

        if airport == destination:

            completed_paths.append((g_score, path))

            if not SHOW_TOP_10_PATHS:

                runtime = time.perf_counter() - start_runtime

                return {
                    "best_cost": g_score,
                    "best_path": path,
                    "nodes_expanded": nodes_expanded,
                    "runtime": runtime,
                }

            if len(completed_paths) >= 10:
                break

        if segments_used >= MAX_SEGMENTS:
            continue

        state_key = (
            airport,
            segments_used,
        )

        if (
            state_key in best_seen
            and best_seen[state_key] <= g_score
        ):
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

                if layover_minutes > DEFAULT_USER_MAX_LAYOVER_HOURS * 60:
                    continue

            else:
                layover_minutes = 0

            new_g = (
                g_score
                + edge_cost(flight, layover_minutes)
            )

            new_path = path + [flight]

            h = heuristic[flight.arrival_airport]

            new_f = new_g + h

            heapq.heappush(
                pq,
                (
                    new_f,
                    new_g,
                    next(counter),
                    flight.arrival_airport,
                    flight.arrival_time,
                    segments_used + 1,
                    new_path,
                )
            )

    runtime = time.perf_counter() - start_runtime

    completed_paths.sort(key=lambda x: x[0])

    best_cost, best_path = completed_paths[0]

    if SHOW_TOP_10_PATHS:

        print("\n==============================")
        print("TOP 10 PATHS")
        print("==============================\n")

        for rank, (cost, path) in enumerate(
            completed_paths[:10],
            start=1,
        ):

            print(f"{rank}. COST = ${cost:.2f}")

            print_path(path)

            print()

    return {
        "best_cost": best_cost,
        "best_path": best_path,
        "nodes_expanded": nodes_expanded,
        "runtime": runtime,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    flights = load_flights("nonstop.db")

    adjacency = build_graph(flights)

    result = astar(
        adjacency=adjacency,
        flights=flights,
        origin="IAD",
        destination="LAX",
    )

    print("\n==============================")
    print("BEST ROUTE")
    print("==============================\n")

    print_path(result["best_path"])

    print("\n==============================")
    print("STATS")
    print("==============================")

    print(f"Best Cost: ${result['best_cost']:.2f}")
    print(f"Nodes Expanded: {result['nodes_expanded']}")
    print(f"Runtime: {result['runtime']:.4f} seconds")