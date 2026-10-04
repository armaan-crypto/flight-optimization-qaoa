"""
qaoa_flight_optimizer.py
========================
Finds the cheapest multi-hop flight route from ORIGIN to DESTINATION.

HOW TO USE
----------
Edit the CONFIG block directly below, then run:
    python qaoa_flight_optimizer.py

Requirements:
    pip install numpy
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← edit everything in this block, then run the file
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH     = "nonstop.db"
ORIGIN      = "ABQ"
DESTINATION = "BZN"

# ── Stop / segment limits ──────────────────────────────────────────────────────
# MAX_SEGMENTS is auto-capped by direct distance if USE_DISTANCE_BASED_SEGMENTS
# is True. The table below is the effective cap:
#   direct < 800km   → 2 segments
#   direct < 2000km  → 3 segments
#   direct >= 2000km → 4 segments (hard max)
MAX_SEGMENTS               = 4      # absolute hard cap — never exceeded
USE_DISTANCE_BASED_SEGMENTS = True  # auto-reduce cap for short routes

# ── Total flight distance cap ──────────────────────────────────────────────────
# Reject any path whose sum of leg distances exceeds this multiple of the
# direct origin→destination distance. Cuts routes like ABQ->DAL->AUS->DEN->BZN
# that accumulate huge total distance even if each stop passes geo pruning.
# Set to None to disable.
MAX_TOTAL_DISTANCE_FACTOR = 2.5

# ── Layover constraints ────────────────────────────────────────────────────────
MIN_LAYOVER_HOURS = 1
MAX_LAYOVER_HOURS = 6

# ── Hub rank pruning ───────────────────────────────────────────────────────────
# Only allow intermediate stops at airports ranked within this position in
# the hub list below. Lower = stricter. e.g. 30 allows only major hubs.
# Set to None to disable (allow any airport as a stop).
MAX_HUB_RANK = 40

# ── Geographic detour pruning ─────────────────────────────────────────────────
# If True, an intermediate stop is only allowed if its perpendicular distance
# to the straight line segment between ORIGIN and DESTINATION is within
# GEO_DETOUR_FACTOR * direct_distance.
#
# This measures how far OFF the corridor a stop is, not how far along it.
# LAX sits ~1050km from the ABQ->BZN line  (factor ~0.84) — allowed at 1.0
# ORD sits ~1760km from the ABQ->BZN line  (factor ~1.40) — cut at 1.0
# HNL sits ~5200km from the ABQ->BZN line  (factor ~4.14) — cut at 1.0
# DEN sits  ~325km from the ABQ->BZN line  (factor ~0.26) — always allowed
#
# Recommended starting value: 1.0 (allows LAX, PHX, SLC, DEN, SMF; cuts ORD/HNL)
USE_GEO_PRUNING   = True
GEO_DETOUR_FACTOR = 1.0   # max perpendicular offset as multiple of direct distance

# ── Airline preferences ────────────────────────────────────────────────────────
# Substrings to match airline names (case-insensitive), e.g. ["Delta", "United"]
PREFERRED_AIRLINES = []   # leave [] for no preference
EXCLUDED_AIRLINES  = []   # leave [] to allow all airlines

# ── Airport avoidance ─────────────────────────────────────────────────────────
# IATA codes to exclude as intermediate stops, e.g. ["ORD", "DFW"]
EXCLUDED_AIRPORTS  = []   # leave [] to allow all airports

# ── Total travel time cap ─────────────────────────────────────────────────────
# Hard cap on elapsed time origin departure → destination arrival (hours).
MAX_TOTAL_TRAVEL_HOURS = None   # e.g. 12.0, or None to disable

# ── Fare projection defaults (keep in sync with astar.py) ────────────────────
DEFAULT_SEATS_REMAINING = 55

# ══════════════════════════════════════════════════════════════════════════════
#  SCORING WEIGHTS  ← keep in sync with astar.py
#
#  Each component is normalized to [0,1] via data-driven reference points,
#  multiplied by its weight, summed/divided by w_sum, then sigmoid → (0,1).
#  Lower score = better route. Weights are relative (no fixed sum needed).
# ══════════════════════════════════════════════════════════════════════════════

WEIGHT_COST     = 3.0   # predicted fare                    (dominant)
WEIGHT_LAYOVER  = 1.0   # deviation from ideal layover
WEIGHT_SEGMENTS = 0.8   # prefer fewer stops
WEIGHT_AIRLINE  = 0.6   # preferred carrier bonus
WEIGHT_DEP_TIME = 0.3   # prefer mid-window departures
WEIGHT_TRAVEL   = 0.3   # prefer shorter flight legs

# Sigmoid shift: average route scores ~0.5. Raise to push scores lower.
SIGMOID_CENTER = 0.5

# Ideal layover — zero layover penalty at exactly this value (hours).
DESIRED_LAYOVER_HOURS = 2.0

# Departure/arrival window (24-hour ints, inclusive).
# Hard filter AND soft scoring reference (ideal = midpoint of window).
DEP_WINDOW_START = 0    # e.g. 6  = reject before 6 AM
DEP_WINDOW_END   = 23   # e.g. 21 = reject after  9 PM
ARR_WINDOW_START = 0
ARR_WINDOW_END   = 23   # e.g. 22 = reject after 10 PM

# ── Output ────────────────────────────────────────────────────────────────────
SHOW_TOP_10_PATHS = True

# ══════════════════════════════════════════════════════════════════════════════
#  END OF CONFIG — no need to edit anything below this line
# ══════════════════════════════════════════════════════════════════════════════

import math
import sqlite3
import time
from collections import defaultdict
from datetime import datetime

from fare_evolution import project_fare

TIME_FMT = "%I:%M %p"

# ── Airport hub rank list ─────────────────────────────────────────────────────
# Position in this list = hub rank (index 0 = busiest). Used to prune
# intermediate stops at obscure airports that no sensible connection uses.
_RAW_AIRPORTS = [
    "ATL","LAX","ORD","DFW","DEN","JFK","SFO","SEA","LAS","MCO",
    "EWR","CLT","PHX","MIA","IAH","BOS","MSP","DTW","FLL","PHL",
    "LGA","BWI","SLC","DCA","IAD","MDW","HNL","SAN","TPA","PDX","STL",
    "BNA","AUS","RDU","MCI","OAK","SJC","SMF","DAL","HOU","MSY",
    "RSW","CLE","IND","CMH","PIT","MKE","OMA","BUF","RIC","ORF",
    "JAX","SAT","ABQ","TUL","OKC","MEM","BHM","SDF","DSM","GRR",
    "ANC","FAI","JNU","TUS","FLG","XNA","LIT","FSM","TXK",
    "BUR","LGB","ONT","PSP","FAT","SBA","SBP","ACV","SNA",
    "MRY","STS","BFL","CLD","CIC","MOD","MMH",
    "COS","GJT","DRO","BDL","HVN","TLH","PNS","GNV","ECP",
    "SAV","AGS","CSG","VLD","ABY","HSV","MOB","MGM","DHN",
    "ITO","OGG","LIH","KOA","BOI","TWF","IDA","PIH","LWS",
    "SPI","BMI","SBN","EVV","FWA","MLI","CID","SUX","ALO","DBQ",
    "ICT","MHK","LEX","BTR","LFT","SHV","MLU","AEX","LCH",
    "BGR","PWM","HGR","SBY","ACK","MVY","ORH","HYA",
    "FNT","LAN","AZO","TVC","MQT","DLH","RST","HIB","BJI","BRD",
    "GPT","JAN","SGF","JLN","BZN","MSO","GTF","BIL","HLN","FCA",
    "LNK","OFK","RNO","MHT","PSM","ACY","SAF",
    "ALB","SYR","ROC","ISP","HPN","BGM","ELM","ITH","SWF",
    "GSO","GSP","AVL","ILM","OAJ","FAY","INT","EWN",
    "FAR","BIS","MOT","GFK","DIK","ISN","DAY","TOL","CAK","LAW",
    "EUG","MFR","RDM","ABE","MDT","AVP","ERI","IPT","PVD",
    "MYR","CAE","CHS","HHH","FSD","RAP","PIR","ATY","ABR",
    "TYS","CHA","MQY","AMA","LBB","MAF","CRP","ELP",
    "GRK","HRL","MFE","BRO","LRD","GGG","TYR","ACT","SJT","SPS","ABI",
    "SGU","PVU","BTV","CHO","ROA","LYH","PHF","GEG",
    "CRW","HTS","CKB","ATW","MSN","LSE","GRB","CWA","EAU","RHI",
    "CPR","JAC","GCC","SJU","BQN","PSE","STT","STX","GUM",
]
HUB_RANK = {code: rank for rank, code in enumerate(_RAW_AIRPORTS)}

# ── Airport coordinates (lat, lon) for geo-distance pruning ──────────────────
AIRPORT_COORDS = {
    "ATL":(33.64,-84.43),"LAX":(33.94,-118.41),"ORD":(41.98,-87.90),
    "DFW":(32.90,-97.04),"DEN":(39.86,-104.67),"JFK":(40.64,-73.78),
    "SFO":(37.62,-122.38),"SEA":(47.45,-122.31),"LAS":(36.08,-115.15),
    "MCO":(28.43,-81.31),"EWR":(40.69,-74.17),"CLT":(35.21,-80.94),
    "PHX":(33.44,-112.01),"MIA":(25.79,-80.29),"IAH":(29.98,-95.34),
    "BOS":(42.36,-71.01),"MSP":(44.88,-93.22),"DTW":(42.21,-83.35),
    "FLL":(26.07,-80.15),"PHL":(39.87,-75.24),"LGA":(40.78,-73.87),
    "BWI":(39.18,-76.67),"SLC":(40.79,-111.98),"DCA":(38.85,-77.04),
    "IAD":(38.94,-77.46),"MDW":(41.79,-87.75),"HNL":(21.32,-157.92),
    "SAN":(32.73,-117.19),"TPA":(27.98,-82.53),"PDX":(45.59,-122.60),
    "STL":(38.75,-90.37),"BNA":(36.12,-86.68),"AUS":(30.20,-97.67),
    "RDU":(35.88,-78.79),"MCI":(39.30,-94.71),"OAK":(37.72,-122.22),
    "SJC":(37.36,-121.93),"SMF":(38.69,-121.59),"DAL":(32.85,-96.85),
    "HOU":(29.65,-95.28),"MSY":(29.99,-90.26),"RSW":(26.54,-81.76),
    "CLE":(41.41,-81.85),"IND":(39.72,-86.29),"CMH":(39.99,-82.89),
    "PIT":(40.49,-80.23),"MKE":(42.95,-87.90),"OMA":(41.30,-95.89),
    "BUF":(42.94,-78.73),"RIC":(37.51,-77.32),"ORF":(36.90,-76.02),
    "JAX":(30.49,-81.69),"SAT":(29.53,-98.47),"ABQ":(35.04,-106.61),
    "TUL":(36.20,-95.89),"OKC":(35.39,-97.60),"MEM":(35.04,-89.98),
    "BHM":(33.56,-86.75),"SDF":(38.17,-85.74),"DSM":(41.53,-93.66),
    "GRR":(42.88,-85.52),"ANC":(61.17,-150.02),"FAI":(64.82,-147.86),
    "JNU":(58.35,-134.57),"TUS":(32.12,-110.94),"FLG":(35.14,-111.67),
    "XNA":(36.28,-94.31),"LIT":(34.73,-92.22),"BOI":(43.56,-116.22),
    "TWF":(42.48,-114.49),"IDA":(43.51,-112.07),"PIH":(42.91,-112.60),
    "LWS":(46.37,-117.01),"COS":(38.81,-104.70),"GJT":(39.12,-108.53),
    "DRO":(37.15,-107.75),"BDL":(41.93,-72.68),"ICT":(37.65,-97.43),
    "BTR":(30.53,-91.15),"SHV":(32.45,-93.82),"BGR":(44.81,-68.83),
    "PWM":(43.64,-70.31),"FNT":(42.97,-83.74),"LAN":(42.77,-84.59),
    "AZO":(42.23,-85.55),"TVC":(44.74,-85.58),"DLH":(46.84,-92.19),
    "RST":(43.91,-92.50),"GPT":(30.41,-89.07),"JAN":(32.31,-90.08),
    "SGF":(37.25,-93.39),"BZN":(45.78,-111.15),"MSO":(46.92,-114.09),
    "GTF":(47.48,-111.37),"BIL":(45.81,-108.54),"HLN":(46.61,-111.98),
    "FCA":(48.31,-114.26),"LNK":(40.85,-96.76),"RNO":(39.50,-119.77),
    "MHT":(42.93,-71.44),"ALB":(42.75,-73.80),"SYR":(43.11,-76.10),
    "ROC":(43.12,-77.67),"FAR":(46.92,-96.81),"BIS":(46.77,-100.75),
    "MOT":(48.26,-101.28),"GFK":(47.95,-97.18),"DAY":(39.90,-84.22),
    "TOL":(41.59,-83.81),"CAK":(40.92,-81.44),"EUG":(44.12,-123.22),
    "MFR":(42.37,-122.87),"RDM":(44.25,-121.15),"PVD":(41.73,-71.43),
    "MYR":(33.68,-78.93),"CAE":(33.94,-81.12),"CHS":(32.90,-80.04),
    "FSD":(43.58,-96.74),"RAP":(44.04,-103.06),"TYS":(35.81,-83.99),
    "CHA":(35.03,-85.20),"AMA":(35.22,-101.71),"LBB":(33.66,-101.82),
    "MAF":(31.94,-102.20),"CRP":(27.77,-97.50),"ELP":(31.81,-106.38),
    "GEG":(47.62,-117.54),"CPR":(42.91,-106.47),"JAC":(43.61,-110.74),
    "GCC":(44.35,-105.54),"SJU":(18.44,-66.00),"BTV":(44.47,-73.15),
    "ROA":(37.32,-79.97),"SAF":(35.61,-106.09),"SGU":(37.09,-113.60),
    "SLC_":(40.79,-111.98),  # alias handled via SLC
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FLIGHT CLASS  (identical to astar.py)
# ══════════════════════════════════════════════════════════════════════════════

def parse_time(raw):
    return datetime.strptime(raw.split(" on")[0], TIME_FMT)


class Flight:

    __slots__ = (
        "id", "airline", "departure_airport", "arrival_airport",
        "departure_time", "arrival_time", "duration_minutes",
        "snapshot_price", "cabin", "is_best_flight",
    )

    def __init__(self, row):
        (
            self.id, self.airline, _flight_date,
            self.departure_airport, self.arrival_airport,
            dep_raw, arr_raw,
            self.duration_minutes, self.snapshot_price,
            self.cabin, self.is_best_flight,
        ) = row
        self.departure_time = parse_time(dep_raw)
        self.arrival_time   = parse_time(arr_raw)

    def __repr__(self):
        return f"{self.departure_airport}->{self.arrival_airport} ({self.airline})"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_flights(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, segmentsAirlineName, flightDate,
               segmentsDepartureAirportCode, segmentsArrivalAirportCode,
               segmentsDepartureTimeRaw, segmentsArrivalTimeRaw,
               travelDuration, totalFare, segmentsCabinCode, isBestFlight
        FROM flights
    """)
    flights = [Flight(row) for row in cur.fetchall()]
    conn.close()
    return flights


def build_graph(flights):
    adjacency = defaultdict(list)
    for flight in flights:
        adjacency[flight.departure_airport].append(flight)
    for airport in adjacency:
        adjacency[airport].sort(key=lambda f: f.departure_time)
    return adjacency


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FARE CACHE  (identical to astar.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CANDIDATE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(code_a, code_b):
    """Straight-line distance in km between two airports using their coordinates."""
    if code_a not in AIRPORT_COORDS or code_b not in AIRPORT_COORDS:
        return None
    lat1, lon1 = AIRPORT_COORDS[code_a]
    lat2, lon2 = AIRPORT_COORDS[code_b]
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _haversine_coords(a, b):
    """Same as _haversine but takes raw (lat, lon) tuples."""
    R = 6371.0
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    h = math.sin(dlat/2)**2 + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(h))


def _perp_distance_to_segment(origin_code, dest_code, stop_code):
    """
    Perpendicular distance (km) from stop to the line segment origin->destination.
    Uses equirectangular projection centred on the midpoint — accurate enough
    for continental distances. Clamps to the nearest endpoint if the foot of
    the perpendicular falls outside the segment.
    Returns None if any airport is missing from AIRPORT_COORDS.
    """
    if any(c not in AIRPORT_COORDS for c in (origin_code, dest_code, stop_code)):
        return None

    origin = AIRPORT_COORDS[origin_code]
    dest   = AIRPORT_COORDS[dest_code]
    stop   = AIRPORT_COORDS[stop_code]

    # equirectangular projection centred on midpoint latitude
    lat0    = math.radians((origin[0] + dest[0]) / 2)
    scale_x = math.cos(lat0)

    def proj(coord):
        return (
            math.radians(coord[1]) * scale_x * 6371.0,
            math.radians(coord[0])             * 6371.0,
        )

    ox, oy = proj(origin)
    dx, dy = proj(dest)
    sx, sy = proj(stop)

    abx, aby = dx - ox, dy - oy
    apx, apy = sx - ox, sy - oy

    ab_len_sq = abx*abx + aby*aby
    if ab_len_sq == 0:
        return _haversine_coords(origin, stop)

    # t = projection parameter, clamped to [0, 1]
    t = max(0.0, min(1.0, (apx*abx + apy*aby) / ab_len_sq))

    foot = (
        origin[0] + t * (dest[0] - origin[0]),
        origin[1] + t * (dest[1] - origin[1]),
    )
    return _haversine_coords(foot, stop)


def build_itinerary_candidates(adjacency):
    """
    Exhaustive DFS from ORIGIN to DESTINATION with two pruning strategies:

    1. Hub rank pruning (MAX_HUB_RANK):
       Intermediate stops must appear within the top MAX_HUB_RANK airports
       in the hub list. Airports outside this rank (e.g. HNL, obscure regionals)
       are skipped as connection points unless they are the destination.

    2. Geo-distance pruning (USE_GEO_PRUNING):
       Intermediate stops must be within GEO_DETOUR_FACTOR * direct_distance
       from the origin. Cuts routes that fly wildly out of direction.

    Both can be disabled in CONFIG.
    Sorted by total predicted fare (cheapest first).
    """
    # pre-compute direct origin→destination distance once
    direct_dist = _haversine(ORIGIN, DESTINATION)

    # auto-cap segment count based on direct distance
    if USE_DISTANCE_BASED_SEGMENTS and direct_dist is not None:
        if direct_dist < 800:
            effective_max_segments = min(2, MAX_SEGMENTS)
        elif direct_dist < 2000:
            effective_max_segments = min(3, MAX_SEGMENTS)
        else:
            effective_max_segments = MAX_SEGMENTS
    else:
        effective_max_segments = MAX_SEGMENTS

    max_total_dist = (direct_dist * MAX_TOTAL_DISTANCE_FACTOR
                      if direct_dist and MAX_TOTAL_DISTANCE_FACTOR else None)

    candidates = []

    def dfs(path, current_airport, current_arrival_time, airports_in_path, total_dist_so_far):
        if len(path) >= effective_max_segments:
            return

        for flight in adjacency.get(current_airport, []):
            arr = flight.arrival_airport
            is_final = (arr == DESTINATION)

            # ── hard filter: excluded airlines ────────────────────────────────
            if EXCLUDED_AIRLINES:
                if any(ex.lower() in flight.airline.lower() for ex in EXCLUDED_AIRLINES):
                    continue

            # ── hard filter: excluded intermediate airports ────────────────────
            if not is_final and EXCLUDED_AIRPORTS:
                if arr in EXCLUDED_AIRPORTS:
                    continue

            # ── hard filter: departure time window ────────────────────────────
            if not (DEP_WINDOW_START <= flight.departure_time.hour <= DEP_WINDOW_END):
                continue

            # ── hard filter: arrival time window ─────────────────────────────
            if not (ARR_WINDOW_START <= flight.arrival_time.hour <= ARR_WINDOW_END):
                continue

            # layover window check (skip for first leg)
            if current_arrival_time is not None:
                layover_min = (
                    flight.departure_time - current_arrival_time
                ).total_seconds() / 60
                if layover_min < MIN_LAYOVER_HOURS * 60:
                    continue
                if layover_min > MAX_LAYOVER_HOURS * 60:
                    continue

            # no cycles
            if arr in airports_in_path:
                continue

            # accumulate leg distance for total path distance cap
            leg_dist = _haversine(current_airport, arr) or 0.0
            new_total_dist = total_dist_so_far + leg_dist
            if max_total_dist is not None and new_total_dist > max_total_dist:
                continue

            # intermediate stop checks (not applied when landing at destination)
            if not is_final:

                # hub rank pruning
                if MAX_HUB_RANK is not None:
                    rank = HUB_RANK.get(arr)
                    if rank is None or rank >= MAX_HUB_RANK:
                        continue

                # geo pruning: reject stops too far off the CURRENT→DESTINATION
                # corridor. As the path progresses the valid zone tightens.
                if USE_GEO_PRUNING:
                    seg_dist = _haversine(current_airport, DESTINATION)
                    if seg_dist is not None and seg_dist > 0:
                        perp = _perp_distance_to_segment(current_airport, DESTINATION, arr)
                        if perp is not None and perp > GEO_DETOUR_FACTOR * seg_dist:
                            continue

            new_path = path + [flight]

            if is_final:
                # total travel time cap
                if MAX_TOTAL_TRAVEL_HOURS is not None:
                    first_dep = new_path[0].departure_time
                    total_hrs = (flight.arrival_time - first_dep).total_seconds() / 3600
                    if total_hrs > MAX_TOTAL_TRAVEL_HOURS:
                        continue
                candidates.append(new_path)
            else:
                dfs(new_path, arr, flight.arrival_time, airports_in_path | {arr}, new_total_dist)

    dfs([], ORIGIN, None, {ORIGIN}, 0.0)

    # sort by weighted score: fare + stop penalty + airline penalty
    candidates.sort(key=itinerary_cost)
    return candidates


# ── NormStats + sigmoid scoring — mirrors astar.py exactly ──────────────────

class NormStats:
    """
    Data-driven reference points for [0,1] normalization.
    0.0 = ideal value for that component; 1.0 = worst observed deviation.
    Computed once after flights load; never hardcoded.
    """
    __slots__ = (
        "fare_ideal", "fare_worst", "dur_ideal", "dur_worst",
        "lay_ideal", "lay_worst", "seg_worst",
        "dep_midpoint", "dep_half_width", "w_sum",
    )
    def __init__(self, fare_ideal, fare_worst, dur_ideal, dur_worst,
                 lay_ideal, lay_worst, seg_worst,
                 dep_midpoint, dep_half_width, w_sum):
        self.fare_ideal = fare_ideal; self.fare_worst = fare_worst
        self.dur_ideal  = dur_ideal;  self.dur_worst  = dur_worst
        self.lay_ideal  = lay_ideal;  self.lay_worst  = lay_worst
        self.seg_worst  = seg_worst
        self.dep_midpoint   = dep_midpoint
        self.dep_half_width = dep_half_width
        self.w_sum = w_sum


def compute_norm_stats(flights):
    """Single pass over all flights. Mirrors compute_norm_stats() in astar.py."""
    fares = [get_predicted_fare(f) for f in flights]
    durs  = [f.duration_minutes    for f in flights]
    fare_min, fare_max = min(fares), max(fares)
    dur_min,  dur_max  = min(durs),  max(durs)

    lay_worst = max(
        MAX_LAYOVER_HOURS - DESIRED_LAYOVER_HOURS,
        DESIRED_LAYOVER_HOURS - MIN_LAYOVER_HOURS,
        0.01,
    )
    dep_midpoint   = (DEP_WINDOW_START + DEP_WINDOW_END) / 2.0
    dep_half_width = max((DEP_WINDOW_END - DEP_WINDOW_START) / 2.0, 0.01)
    w_sum = (WEIGHT_COST + WEIGHT_LAYOVER + WEIGHT_SEGMENTS
             + WEIGHT_AIRLINE + WEIGHT_DEP_TIME + WEIGHT_TRAVEL)

    return NormStats(
        fare_ideal     = fare_min,
        fare_worst     = max(fare_max - fare_min, 1.0),
        dur_ideal      = dur_min,
        dur_worst      = max(dur_max - dur_min, 1.0),
        lay_ideal      = DESIRED_LAYOVER_HOURS,
        lay_worst      = lay_worst,
        seg_worst      = float(max(MAX_SEGMENTS - 1, 1)),
        dep_midpoint   = dep_midpoint,
        dep_half_width = dep_half_width,
        w_sum          = w_sum,
    )


def _refpoint(value, ideal, worst_deviation):
    """0.0 = at ideal; 1.0 = worst_deviation away. Clamped to [0,1]."""
    return max(0.0, min(1.0, abs(value - ideal) / worst_deviation))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# Module-level NormStats — set in __main__ after flights load.
_ns = None


def _leg_score(flight, layover_minutes, segments_used):
    """
    Normalized, weighted, sigmoid score for one leg. Mirrors edge_cost()
    in astar.py. All six components use the same weights and formula.
    Lower = better.
    """
    pen_cost     = _refpoint(get_predicted_fare(flight),
                             _ns.fare_ideal, _ns.fare_worst)
    pen_layover  = (_refpoint(layover_minutes / 60.0,
                              _ns.lay_ideal, _ns.lay_worst)
                    if segments_used > 0 else 0.0)
    pen_segments = _refpoint(float(segments_used), 0.0, _ns.seg_worst)
    pen_airline  = (0.0 if not PREFERRED_AIRLINES else
                    0.0 if any(p.lower() in flight.airline.lower()
                               for p in PREFERRED_AIRLINES) else 1.0)
    pen_dep      = _refpoint(float(flight.departure_time.hour),
                             _ns.dep_midpoint, _ns.dep_half_width)
    pen_travel   = _refpoint(flight.duration_minutes,
                             _ns.dur_ideal, _ns.dur_worst)

    weighted_sum = (
        WEIGHT_COST     * pen_cost
        + WEIGHT_LAYOVER  * pen_layover
        + WEIGHT_SEGMENTS * pen_segments
        + WEIGHT_AIRLINE  * pen_airline
        + WEIGHT_DEP_TIME * pen_dep
        + WEIGHT_TRAVEL   * pen_travel
    ) / _ns.w_sum

    return _sigmoid(weighted_sum - SIGMOID_CENTER)


def itinerary_cost(path):
    """
    Cumulative sigmoid score for a complete itinerary — sum of per-leg scores.
    Mirrors g_score accumulation in astar.py. Lower = better.
    Used both to rank candidates and to build the QUBO cost diagonal.
    """
    total = 0.0
    for i, flight in enumerate(path):
        layover_min = 0.0
        if i > 0:
            layover_min = (
                flight.departure_time - path[i - 1].arrival_time
            ).total_seconds() / 60
        total += _leg_score(flight, layover_min, segments_used=i)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 5.  OUTPUT  (identical format to astar.py)
# ══════════════════════════════════════════════════════════════════════════════

def format_minutes(minutes):
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h}h {m:02d}m"


def print_path(path):
    if not path:
        return

    route = [path[0].departure_airport] + [f.arrival_airport for f in path]
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


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    start_runtime = time.perf_counter()

    print(f"\n[1/2] Loading flights from {DB_PATH}…")
    flights   = load_flights(DB_PATH)
    adjacency = build_graph(flights)
    print(f"      {len(flights)} flights loaded.")

    # build normalization reference points from data — mirrors astar.py
    _ns = compute_norm_stats(flights)


    print(f"[2/2] Finding cheapest routes ({ORIGIN} → {DESTINATION})…")
    candidates = build_itinerary_candidates(adjacency)
    print(f"      {len(candidates)} feasible itineraries found.")

    if not candidates:
        print("  No feasible routes found. Relax constraints and retry.")
        raise SystemExit(1)

    # ── dump all feasible itineraries to file ────────────────────────────────
    dump_path = "all_itineraries.txt"
    with open(dump_path, "w") as dump:
        dump.write(f"ALL FEASIBLE ITINERARIES: {ORIGIN} -> {DESTINATION}\n")
        dump.write(f"Total found: {len(candidates)}\n")
        dump.write("=" * 60 + "\n\n")
        for rank, path in enumerate(candidates, start=1):
            fare = sum(get_predicted_fare(fl) for fl in path)
            route = " -> ".join(
                [path[0].departure_airport] + [fl.arrival_airport for fl in path]
            )
            dump.write(f"{rank}. TOTAL FARE = ${fare:.2f}  |  {route}\n")
            for i, fl in enumerate(path):
                dep_str = fl.departure_time.strftime("%I:%M %p").lstrip("0")
                arr_str = fl.arrival_time.strftime("%I:%M %p").lstrip("0")
                dump.write(
                    f"   Leg {i+1}: {fl.departure_airport} -> {fl.arrival_airport}"
                    f" | {fl.airline}"
                    f" | departs {dep_str} arrives {arr_str}"
                    f" | ${get_predicted_fare(fl):.2f}"
                    f" | {format_minutes(fl.duration_minutes)}\n"
                )
                if i > 0:
                    layover_min = (
                        fl.departure_time - path[i - 1].arrival_time
                    ).total_seconds() / 60
                    dump.write(f"   Layover at {fl.departure_airport}: {format_minutes(layover_min)}\n")
            dump.write("\n")
    print(f"      All itineraries written to {dump_path}")

    runtime = time.perf_counter() - start_runtime

    if SHOW_TOP_10_PATHS:
        print("\n==============================")
        print("TOP 10 PATHS")
        print("==============================\n")
        for rank, path in enumerate(candidates[:10], start=1):
            fare = sum(get_predicted_fare(f) for f in path)
            print(f"{rank}. TOTAL FARE = ${fare:.2f}")
            print_path(path)
            print()

    best = candidates[0]
    best_fare = sum(get_predicted_fare(f) for f in best)

    print("\n==============================")
    print("BEST ROUTE")
    print("==============================\n")
    print_path(best)

    print("\n==============================")
    print("STATS")
    print("==============================")
    print(f"Total Fare:     ${best_fare:.2f}")
    print(f"Routes Found:   {len(candidates)}")
    print(f"Runtime:        {runtime:.4f} seconds")