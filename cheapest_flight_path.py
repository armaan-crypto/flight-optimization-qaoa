import os
import re
import time
import sqlite3
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


DB_FILENAME = "flights.db"


def get_db_path() -> str:
    """
    Resolve the path to flights.db, assuming it lives alongside this script.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DB_FILENAME)


def parse_fare(fare_str: str) -> Optional[float]:
    """
    Try to extract a numeric fare from the totalFare TEXT column.
    Handles values like '123.45', '$123.45', 'USD 123.45', etc.
    Returns None if no usable number is found.
    """
    if fare_str is None:
        return None

    fare_str = fare_str.strip()

    # Fast path: plain float
    try:
        return float(fare_str)
    except ValueError:
        pass

    # Fallback: grab the first number (integer or decimal) from the string
    match = re.search(r"(\d+(\.\d+)?)", fare_str)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def build_graph(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, float, int]]]:
    """
    Build an adjacency list graph from the flights table.

    Nodes: airport codes (origin / dest).
    Edges: directed flights origin -> dest with weight = fare.
    """
    cur = conn.cursor()
    # Ignore flights with zero remaining seats if that indicates unavailability.
    cur.execute(
        """
        SELECT id, origin, dest, totalFare, seatsRemaining
        FROM flights
        """
    )

    graph: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)

    for flight_id, origin, dest, total_fare, seats_remaining in cur.fetchall():
        # Optionally skip flights with no remaining seats
        if str(seats_remaining).strip() in {"0", "0.0"}:
            continue

        cost = parse_fare(total_fare)
        if cost is None:
            continue

        graph[str(origin).upper()].append((str(dest).upper(), cost, int(flight_id)))

    return graph


def dijkstra(
    graph: Dict[str, List[Tuple[str, float, int]]], start: str, goal: str
) -> Tuple[Optional[float], Optional[List[str]], Optional[List[int]]]:
    """
    Dijkstra's algorithm for the cheapest path (by total fare).

    Returns (total_cost, path_as_list_of_airports, path_as_list_of_flight_ids)
    or (None, None, None) if unreachable.
    """
    if start == goal:
        return 0.0, [start], []

    # priority queue of (accumulated_cost, node, path_airports, path_flight_ids)
    pq: List[Tuple[float, str, List[str], List[int]]] = [(0.0, start, [start], [])]
    best_cost: Dict[str, float] = {start: 0.0}

    while pq:
        cost, node, path_airports, path_flights = heapq.heappop(pq)

        # If we've already found a better route to this node, skip
        if cost > best_cost.get(node, float("inf")):
            continue

        if node == goal:
            return cost, path_airports, path_flights

        for neighbor, weight, flight_id in graph.get(node, []):
            new_cost = cost + weight
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                heapq.heappush(
                    pq,
                    (new_cost, neighbor, path_airports + [neighbor], path_flights + [flight_id]),
                )

    return None, None, None


def main() -> None:
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"Could not find '{DB_FILENAME}' next to this script.")
        return

    start = input("Enter starting airport code (e.g. SFO): ").strip().upper()
    goal = input("Enter destination airport code (e.g. JFK): ").strip().upper()

    if not start or not goal:
        print("Both starting and destination airport codes are required.")
        return

    # Time everything from DB query + graph build through path search
    t0 = time.time()

    with sqlite3.connect(db_path) as conn:
        graph = build_graph(conn)

    if start not in graph:
        print(f"No outgoing flights found from '{start}'.")
        return

    # Quick reachability sanity check: does any edge end at goal?
    reachable_destinations = {dest for edges in graph.values() for dest, _, _ in edges}
    if goal not in reachable_destinations:
        print(f"No flights found arriving at '{goal}'.")
        return

    total_cost, path_airports, path_flights = dijkstra(graph, start, goal)

    t1 = time.time()

    if path_airports is None:
        print(f"No route found from {start} to {goal}.")
        return

    print("\nCheapest route (by total fare):")
    print(" -> ".join(path_airports))
    print(f"Total cost: {total_cost:.2f}")
    if path_flights:
        print("Flight IDs:", " -> ".join(str(fid) for fid in path_flights))
    print(f"Runtime: {t1 - t0:.4f} seconds")


if __name__ == "__main__":
    main()

