#!/usr/bin/env python3
"""
rescan.py — Re-scan (origin, dest) pairs from searchLog that returned 0 flights,
            but only where a major hub airport is involved.

Usage:
    python rescan.py --date 2026-06-01
    python rescan.py --date 2026-06-01 --workers 20
    python rescan.py --date 2026-06-01 --workers 20 --resume

Output:
    newscannedflights_<date>.csv   — same schema as flights_<date>.csv
    rescanlog_<date>.csv           — same schema as searchLog_<date>.csv

The script reads searchLog_<date>.csv to find every (origin, dest) pair that
scraped 0 rows and whose origin OR destination is in RESCAN_AIRPORTS (the top-30
hub set).  It prints the total before starting, then scans in parallel.
"""

import argparse
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from fast_flights import FlightData, Passengers, get_flights
except ImportError:
    print("Error: fast-flights is not installed.  Run: pip install fast-flights")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is not installed.  Run: pip install tqdm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Hub airports — only pairs involving these are worth rescanning
# ---------------------------------------------------------------------------

RESCAN_AIRPORTS = {
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "MIA", "IAH", "BOS", "MSP", "DTW", "FLL", "PHL",
    "LGA", "BWI", "SLC", "DCA", "IAD", "MDW", "HNL", "SAN", "TPA", "PDX",
    "STL", "SNA",
}

MAX_RETRIES   = 3    # rescan gets more attempts than the original run
RETRY_BACKOFF = 1.5  # slightly longer back-off too


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

def new_flights_path(date: str) -> Path:
    return Path(f"newscannedflights_{date}.csv")

def rescan_log_path(date: str) -> Path:
    return Path(f"rescanlog_{date}.csv")

def search_log_path(date: str) -> Path:
    return Path(f"searchLog_{date}.csv")

def checkpoint_path(date: str) -> Path:
    return Path(f".rescan_completed_{date}.json")


# ---------------------------------------------------------------------------
# CSV schemas  (identical to flight_scraper.py)
# ---------------------------------------------------------------------------

FLIGHT_FIELDS = [
    "id",
    "segmentsAirlineName",
    "flightDate",
    "segmentsDepartureAirportCode",
    "segmentsArrivalAirportCode",
    "segmentsDepartureTimeRaw",
    "segmentsArrivalTimeRaw",
    "travelDuration",
    "totalFare",
    "segmentsCabinCode",
    "isBestFlight",
]

RESCAN_LOG_FIELDS = ["origin", "destination", "flightDate", "rowsScraped"]


# ---------------------------------------------------------------------------
# Build candidate pairs from searchLog
# ---------------------------------------------------------------------------

def load_zero_hub_pairs(date: str) -> list[tuple[str, str]]:
    """
    Read searchLog_<date>.csv and return every (origin, dest) pair where:
      - rowsScraped == 0
      - origin OR dest is in RESCAN_AIRPORTS
    """
    slp = search_log_path(date)
    if not slp.exists():
        print(f"Error: {slp} not found. Run flight_scraper.py first.")
        sys.exit(1)

    pairs = []
    with open(slp, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows_scraped = int(row.get("rowsScraped", 1))
            except ValueError:
                continue
            if rows_scraped > 0:
                continue
            origin = row.get("origin", "").strip()
            dest   = row.get("destination", "").strip()
            if not origin or not dest:
                continue
            if origin in RESCAN_AIRPORTS or dest in RESCAN_AIRPORTS:
                pairs.append((origin, dest))

    return pairs


# ---------------------------------------------------------------------------
# Checkpoint helpers — thread-safe
# ---------------------------------------------------------------------------

_checkpoint_lock  = threading.Lock()
_completed_shared: set = set()
_checkpoint_dirty: int = 0
CHECKPOINT_EVERY  = 50


def load_completed(date: str) -> set:
    cp = checkpoint_path(date)
    if cp.exists():
        with open(cp) as f:
            return set(tuple(p) for p in json.load(f))
    return set()


def load_completed_from_rescan_log(date: str) -> set:
    """Read rescanlog to rebuild completed set (used by --resume when checkpoint is gone)."""
    rlp = rescan_log_path(date)
    if not rlp.exists():
        return set()
    done = set()
    with open(rlp, newline="") as f:
        for row in csv.DictReader(f):
            o = row.get("origin", "").strip()
            d = row.get("destination", "").strip()
            if o and d:
                done.add((o, d))
    return done


def mark_completed(date: str, pair: tuple) -> None:
    global _checkpoint_dirty
    with _checkpoint_lock:
        _completed_shared.add(pair)
        _checkpoint_dirty += 1
        if _checkpoint_dirty >= CHECKPOINT_EVERY:
            _write_checkpoint(date)
            _checkpoint_dirty = 0


def _write_checkpoint(date: str) -> None:
    with open(checkpoint_path(date), "w") as f:
        json.dump(list(_completed_shared), f)


def flush_checkpoint(date: str) -> None:
    with _checkpoint_lock:
        _write_checkpoint(date)


# ---------------------------------------------------------------------------
# Auto-increment ID — thread-safe, continues from highest existing id
# ---------------------------------------------------------------------------

_id_lock    = threading.Lock()
_id_counter = 0


def init_id_counter(start: int) -> None:
    global _id_counter
    with _id_lock:
        _id_counter = start


def next_id() -> int:
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return _id_counter


def highest_existing_id(date: str) -> int:
    """
    Scan newscannedflights CSV for the highest id so far.
    Also checks the original flights CSV so ids never collide across files.
    """
    max_id = 0
    for path in [new_flights_path(date), Path(f"flights_{date}.csv")]:
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    max_id = max(max_id, int(row.get("id") or 0))
                except (ValueError, TypeError):
                    pass
    return max_id


# ---------------------------------------------------------------------------
# Parsers  (identical to flight_scraper.py)
# ---------------------------------------------------------------------------

def parse_duration(raw: str) -> str:
    if not raw:
        return ""
    h = re.search(r"(\d+)\s*h", raw)
    m = re.search(r"(\d+)\s*m", raw)
    hrs  = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0
    return f"{hrs}:{mins:02d}" if (hrs or mins) else ""


def parse_price(price) -> float | None:
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price) if price > 0 else None
    match = re.search(r"\d+\.?\d*", str(price).replace(",", ""))
    if not match:
        return None
    val = float(match.group())
    return val if val > 0 else None


def is_valid(row: dict) -> bool:
    return bool(
        row["segmentsAirlineName"]
        and row["segmentsDepartureTimeRaw"]
        and row["segmentsArrivalTimeRaw"]
        and row["travelDuration"]
        and row["totalFare"] is not None
    )


def _departure_matches_date(departure_raw: str, date: str) -> bool:
    if not departure_raw:
        return True
    if date in departure_raw:
        return True
    try:
        target = datetime.strptime(date, "%Y-%m-%d")
        patterns = [
            target.strftime("%b %d"),
            target.strftime("%b %-d"),
            target.strftime("%m/%d"),
            target.strftime("%-m/%-d"),
        ]
        return any(p in departure_raw for p in patterns)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Fetch — pure function, safe for threading
# ---------------------------------------------------------------------------

def _call_api(origin: str, dest: str, date: str) -> list:
    try:
        result = get_flights(
            flight_data=[FlightData(date=date, from_airport=origin, to_airport=dest)],
            trip="one-way",
            seat="economy",
            passengers=Passengers(adults=1),
            max_stops=0,
        )
        return result.flights if result and result.flights else []
    except Exception:
        return []


def fetch_pair(origin: str, dest: str, date: str, delay: float) -> list[dict]:
    """
    Retry up to MAX_RETRIES times (all pairs here are hub pairs, so always retry).
    Returns validated, fully-populated row dicts with IDs assigned.
    """
    flights: list = []
    for attempt in range(MAX_RETRIES + 1):
        flights = _call_api(origin, dest, date)
        if flights:
            break
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)

    if delay > 0:
        time.sleep(delay)

    rows = []
    for flight in flights:
        if getattr(flight, "stops", 0) not in (0, None, ""):
            continue

        departure_raw = str(getattr(flight, "departure", "") or "").strip()
        if not _departure_matches_date(departure_raw, date):
            continue

        candidate = {
            "segmentsAirlineName":          str(getattr(flight, "name",      "") or "").strip(),
            "flightDate":                   date,
            "segmentsDepartureAirportCode": origin,
            "segmentsArrivalAirportCode":   dest,
            "segmentsDepartureTimeRaw":     departure_raw,
            "segmentsArrivalTimeRaw":       str(getattr(flight, "arrival",   "") or "").strip(),
            "travelDuration":               parse_duration(str(getattr(flight, "duration", "") or "")),
            "totalFare":                    parse_price(getattr(flight, "price", None)),
            "segmentsCabinCode":            "ECONOMY",
            "isBestFlight":                 getattr(flight, "is_best", False),
        }
        if is_valid(candidate):
            candidate["id"] = next_id()
            rows.append(candidate)

    return rows


# ---------------------------------------------------------------------------
# Thread-safe CSV writer
# ---------------------------------------------------------------------------

_csv_lock = threading.Lock()


def write_result(
    flight_writer, flight_file,
    log_writer,    log_file,
    origin: str, dest: str, date: str,
    rows: list[dict],
) -> None:
    with _csv_lock:
        for row in rows:
            flight_writer.writerow({f: row[f] for f in FLIGHT_FIELDS})
        log_writer.writerow({
            "origin":      origin,
            "destination": dest,
            "flightDate":  date,
            "rowsScraped": len(rows),
        })
        flight_file.flush()
        log_file.flush()


# ---------------------------------------------------------------------------
# CSV initialisation
# ---------------------------------------------------------------------------

def ensure_csvs(date: str) -> int:
    fp = new_flights_path(date)
    max_id = highest_existing_id(date)
    if not fp.exists():
        with open(fp, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FLIGHT_FIELDS).writeheader()

    rlp = rescan_log_path(date)
    if not rlp.exists():
        with open(rlp, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=RESCAN_LOG_FIELDS).writeheader()

    return max_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-scan zero-result hub pairs from a previous searchLog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--date", required=True,
                        help="Date that was scraped, YYYY-MM-DD (e.g. 2026-06-01)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs already recorded in rescanlog_<date>.csv.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent requests. Try 10-20; lower if errors spike.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Per-worker sleep (s) after each request.")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: --date must be YYYY-MM-DD, got: {args.date}")
        sys.exit(1)

    # Build candidate list from searchLog
    all_pairs = load_zero_hub_pairs(args.date)

    if not all_pairs:
        print("No zero-result hub pairs found in searchLog. Nothing to rescan.")
        sys.exit(0)

    # Apply resume filter
    if args.resume:
        prior = load_completed(args.date) | load_completed_from_rescan_log(args.date)
        _completed_shared.update(prior)
        pairs_to_run = [p for p in all_pairs if p not in _completed_shared]
        print(f"Resuming — skipping {len(prior):,} already-rescanned pairs")
    else:
        pairs_to_run = all_pairs

    max_id = ensure_csvs(args.date)
    init_id_counter(max_id)

    est_min = len(pairs_to_run) / max(args.workers, 1) * max(args.delay, 0.3) / 60
    print(f"Zero-result hub pairs in searchLog : {len(all_pairs):,}")
    print(f"Pairs to rescan                    : {len(pairs_to_run):,}")
    print(f"Workers                            : {args.workers}")
    print(f"Per-worker delay                   : {args.delay}s")
    print(f"Rough ETA                          : ~{est_min:.0f} min")
    print(f"Output flights CSV                 : {new_flights_path(args.date)}")
    print(f"Rescan log CSV                     : {rescan_log_path(args.date)}")
    print()

    total_found  = 0
    still_empty  = 0

    with (
        open(new_flights_path(args.date), "a", newline="") as flight_file,
        open(rescan_log_path(args.date),  "a", newline="") as log_file,
    ):
        flight_writer = csv.DictWriter(flight_file, fieldnames=FLIGHT_FIELDS)
        log_writer    = csv.DictWriter(log_file,    fieldnames=RESCAN_LOG_FIELDS)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_pair, o, d, args.date, args.delay): (o, d)
                for o, d in pairs_to_run
            }

            with tqdm(total=len(pairs_to_run), unit="pair", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    origin, dest = futures[future]
                    try:
                        rows = future.result()
                    except Exception:
                        rows = []

                    write_result(
                        flight_writer, flight_file,
                        log_writer,    log_file,
                        origin, dest, args.date, rows,
                    )
                    mark_completed(args.date, (origin, dest))

                    n = len(rows)
                    total_found += n
                    if n == 0:
                        still_empty += 1

                    pbar.set_postfix(found=total_found, still_empty=still_empty, refresh=False)
                    pbar.update(1)

    flush_checkpoint(args.date)

    print()
    print("Done.")
    print(f"  New flight rows recovered : {total_found:,}  ->  {new_flights_path(args.date)}")
    print(f"  Rescan log entries        : {len(pairs_to_run):,}  ->  {rescan_log_path(args.date)}")
    print(f"  Still empty after rescan  : {still_empty:,} "
          f"({100 * still_empty / max(len(pairs_to_run), 1):.1f}%"
          f" — these routes likely have no nonstop service)")


if __name__ == "__main__":
    main()