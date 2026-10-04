import sqlite3
import heapq
import math
from datetime import datetime

conn = sqlite3.connect("flights.db")
conn.row_factory = sqlite3.Row


# ----------------------------
# Airport coordinates
# ----------------------------
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4279),
    "LAX": (33.9416, -118.409),
    "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.85, -104.674),
    "JFK": (40.6413, -73.7781),
    "SFO": (37.7749, -122.419),
    "MIA": (25.7959, -80.287),
    "CLT": (35.214, -80.9431),
    "EWR": (40.6895, -74.1745),

    # Approximations for ones not in table
    "IAD": (38.9531, -77.4565),
    "OAK": (37.7213, -122.221),
    "BOS": (42.3656, -71.0096),
    "LGA": (40.7769, -73.874),
    "PHL": (39.8744, -75.2424),
    "DTW": (42.2162, -83.3554),
}


# ----------------------------
# Haversine distance heuristic
# ----------------------------
def haversine(a, b):
    lat1, lon1 = AIRPORT_COORDS[a]
    lat2, lon2 = AIRPORT_COORDS[b]

    R = 6371  # km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1-x))


def heuristic(curr, goal):
    if curr not in AIRPORT_COORDS or goal not in AIRPORT_COORDS:
        return 0
    return haversine(curr, goal)


# ----------------------------
# Cost function (you can tune)
# ----------------------------
def cost(flight_row):
    # Example: ticket price
    try:
        return float(flight_row["totalFare"])
    except:
        return 1000


# ----------------------------
# Load graph once (FAST)
# ----------------------------
def load_flights():
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM flights
        ORDER BY departure_time
    """)

    graph = {}
    for row in cur:
        graph.setdefault(row["origin"], []).append(row)

    return graph


# ----------------------------
# Time helpers
# ----------------------------
def parse_time(t):
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


# ----------------------------
# A* Search
# ----------------------------
def astar(start, goal, start_time=None):
    if start_time and start_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=None)

    graph = load_flights()

    pq = []
    heapq.heappush(pq, (0, start, start_time, [], 0))

    best = {}

    while pq:
        f, airport, curr_time, path, g = heapq.heappop(pq)

        if airport == goal:
            return path, g

        state_key = (airport, curr_time)
        if state_key in best and best[state_key] <= g:
            continue
        best[state_key] = g

        for flight in graph.get(airport, []):

            dep = parse_time(flight["departure_time"])
            arr = parse_time(flight["arrival_time"])

            if curr_time and dep < curr_time:
                continue

            step = cost(flight)
            g2 = g + step

            h = heuristic(flight["dest"], goal)
            f2 = g2 + h

            heapq.heappush(
                pq,
                (
                    f2,
                    flight["dest"],
                    arr,
                    path + [flight],
                    g2
                )
            )

    return None, math.inf


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    path, total = astar(
        start="IAD",
        goal="LAX",
        start_time=datetime.fromisoformat("2024-06-01T06:00:00")
    )

    print("TOTAL COST:", total)
    if path:
        for p in path:
            print(p["origin"], "->", p["dest"], p["departure_time"])
