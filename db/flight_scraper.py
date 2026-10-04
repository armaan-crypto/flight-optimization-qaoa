#!/usr/bin/env python3
"""
flight_scraper.py — Scrape domestic US flight data using the fast-flights library.

Usage:
    python flight_scraper.py --date 2026-06-01
    python flight_scraper.py --date 2026-06-01 --workers 20 --resume

Workers: each worker fires an independent HTTP request. Start at 10-20 and
back off (or add --delay 0.5) if you see a spike in empty results, which
usually means Google is rate-limiting the scraper.
"""

import argparse
import csv
import itertools
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
# Airport list -- 130+ major and regional US domestic airports
# ---------------------------------------------------------------------------
_RAW_AIRPORTS = [
    # Top busiest + major hubs
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "MIA", "IAH", "BOS", "MSP", "DTW", "FLL", "PHL",
    "LGA", "BWI", "SLC", "DCA", "IAD", "MDW", "HNL", "SAN", "TPA", "PDX", "STL",
    # Major secondary hubs
    "BNA", "AUS", "RDU", "MCI", "OAK", "SJC", "SMF", "DAL", "HOU", "MSY",
    "RSW", "CLE", "IND", "CMH", "PIT", "MKE", "OMA", "BUF", "RIC", "ORF",
    "JAX", "SAT", "ABQ", "TUL", "OKC", "MEM", "BHM", "SDF", "DSM", "GRR",
    # Alaska
    "ANC", "FAI", "JNU",
    # Arizona
    "TUS", "FLG",
    # Arkansas
    "XNA", "LIT", "FSM", "TXK",
    # California
    "BUR", "LGB", "ONT", "PSP", "FAT", "SBA", "SBP", "ACV", "SNA",
    "MRY", "STS", "BFL", "CLD", "CIC", "MOD", "MMH",
    # Colorado
    "COS", "GJT", "DRO",
    # Connecticut
    "BDL", "HVN",
    # Florida
    "TLH", "PNS", "GNV", "ECP",
    # Georgia
    "SAV", "AGS", "CSG", "VLD", "ABY",
    # Alabama
    "HSV", "MOB", "MGM", "DHN",
    # Hawaii
    "ITO", "OGG", "LIH", "KOA",
    # Idaho
    "BOI", "TWF", "IDA", "PIH", "LWS",
    # Illinois
    "SPI", "BMI",
    # Indiana
    "SBN", "EVV", "FWA",
    # Iowa
    "MLI", "CID", "SUX", "ALO", "DBQ",
    # Kansas
    "ICT", "MHK",
    # Kentucky
    "LEX",
    # Louisiana
    "BTR", "LFT", "SHV", "MLU", "AEX", "LCH",
    # Maine
    "BGR", "PWM",
    # Maryland
    "HGR", "SBY",
    # Massachusetts
    "ACK", "MVY", "ORH", "HYA",
    # Michigan
    "FNT", "LAN", "AZO", "TVC", "MQT",
    # Minnesota
    "DLH", "RST", "HIB", "BJI", "BRD",
    # Mississippi
    "GPT", "JAN",
    # Missouri
    "SGF", "JLN",
    # Montana
    "BZN", "MSO", "GTF", "BIL", "HLN", "FCA",
    # Nebraska
    "LNK", "OFK",
    # Nevada
    "RNO",
    # New Hampshire
    "MHT", "PSM",
    # New Jersey
    "ACY",
    # New Mexico
    "SAF",
    # New York
    "ALB", "SYR", "ROC", "ISP", "HPN", "BGM", "ELM", "ITH", "SWF",
    # North Carolina
    "GSO", "GSP", "AVL", "ILM", "OAJ", "FAY", "INT", "EWN",
    # North Dakota
    "FAR", "BIS", "MOT", "GFK", "DIK", "ISN",
    # Ohio
    "DAY", "TOL", "CAK",
    # Oklahoma
    "LAW",
    # Oregon
    "EUG", "MFR", "RDM",
    # Pennsylvania
    "ABE", "MDT", "AVP", "ERI", "IPT",
    # Rhode Island
    "PVD",
    # South Carolina
    "MYR", "CAE", "CHS", "HHH",
    # South Dakota
    "FSD", "RAP", "PIR", "ATY", "ABR",
    # Tennessee
    "TYS", "CHA", "MQY",
    # Texas
    "AMA", "LBB", "MAF", "CRP", "ELP",
    "GRK", "HRL", "MFE", "BRO", "LRD",
    "GGG", "TYR", "ACT", "SJT", "SPS", "ABI",
    # Utah
    "SGU", "PVU",
    # Vermont
    "BTV",
    # Virginia
    "CHO", "ROA", "LYH", "PHF",
    # Washington
    "GEG",
    # West Virginia
    "CRW", "HTS", "CKB",
    # Wisconsin
    "ATW", "MSN", "LSE", "GRB", "CWA", "EAU", "RHI",
    # Wyoming
    "CPR", "JAC", "GCC",
    # Puerto Rico
    "SJU", "BQN", "PSE",
    # US Virgin Islands
    "STT", "STX",
    # Guam
    "GUM",
]

# Deduplicate while preserving order
_seen_ap: set = set()
AIRPORTS: list = []
for _ap in _RAW_AIRPORTS:
    if _ap not in _seen_ap:
        _seen_ap.add(_ap)
        AIRPORTS.append(_ap)


# ---------------------------------------------------------------------------
# CSV schemas
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

SEARCH_LOG_FIELDS = ["origin", "destination", "flightDate", "rowsScraped"]


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

def flight_csv_path(date: str) -> Path:
    return Path(f"flights_{date}.csv")


def search_log_path(date: str) -> Path:
    return Path(f"searchLog_{date}.csv")


def checkpoint_path(date: str) -> Path:
    return Path(f".completed_{date}.json")


# ---------------------------------------------------------------------------
# Checkpoint helpers -- thread-safe
# ---------------------------------------------------------------------------

_checkpoint_lock = threading.Lock()
_completed_shared: set = set()
_checkpoint_dirty: int = 0
CHECKPOINT_EVERY = 50


def load_completed(date: str) -> set:
    cp = checkpoint_path(date)
    if cp.exists():
        with open(cp) as f:
            return set(tuple(p) for p in json.load(f))
    return set()


def load_completed_from_search_log(date: str) -> set:
    """
    Build the completed-pairs set by reading the searchLog CSV instead of
    (or in addition to) the binary checkpoint file.  This is what --continue
    uses so that a previous full run (whose checkpoint may be gone) can be
    resumed purely from the human-readable log.
    """
    slp = search_log_path(date)
    if not slp.exists():
        return set()
    completed = set()
    with open(slp, newline="") as f:
        for row in csv.DictReader(f):
            o = row.get("origin", "").strip()
            d = row.get("destination", "").strip()
            if o and d:
                completed.add((o, d))
    return completed


def mark_completed(date: str, pair: tuple) -> None:
    global _checkpoint_dirty
    with _checkpoint_lock:
        _completed_shared.add(pair)
        _checkpoint_dirty += 1
        if _checkpoint_dirty >= CHECKPOINT_EVERY:
            _write_checkpoint(date)
            _checkpoint_dirty = 0


def _write_checkpoint(date: str) -> None:
    """Must be called with _checkpoint_lock held."""
    with open(checkpoint_path(date), "w") as f:
        json.dump(list(_completed_shared), f)


def flush_checkpoint(date: str) -> None:
    with _checkpoint_lock:
        _write_checkpoint(date)


# ---------------------------------------------------------------------------
# Auto-increment ID -- thread-safe
# ---------------------------------------------------------------------------

_id_lock = threading.Lock()
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


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_duration(raw: str) -> str:
    """'2 hr 35 min' -> '2:35'.  Returns '' if nothing parseable."""
    if not raw:
        return ""
    h = re.search(r"(\d+)\s*h", raw)
    m = re.search(r"(\d+)\s*m", raw)
    hrs  = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0
    return f"{hrs}:{mins:02d}" if (hrs or mins) else ""


def parse_price(price) -> float | None:
    """Returns a positive float, or None if missing / zero / unparseable."""
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
    """
    Keep a row only when ALL five core fields are present and non-empty.
    This drops the 'price but nothing else' garbage rows the library sometimes
    returns for unsupported or partially-scraped routes.
    """
    return bool(
        row["segmentsAirlineName"]
        and row["segmentsDepartureTimeRaw"]
        and row["segmentsArrivalTimeRaw"]
        and row["travelDuration"]
        and row["totalFare"] is not None
    )


# ---------------------------------------------------------------------------
# Hub airports -- pairs involving these get retried on empty results
# ---------------------------------------------------------------------------

HUB_AIRPORTS = {
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "MIA", "IAH", "BOS", "MSP", "DTW", "FLL", "PHL",
    "LGA", "BWI", "SLC", "DCA", "IAD", "MDW", "HNL", "SAN", "TPA", "PDX",
    "STL", "SNA",
}

MAX_RETRIES   = 2   # extra attempts after the first (hub pairs only)
RETRY_BACKOFF = 1.0 # seconds to wait between retries


# ---------------------------------------------------------------------------
# Fetch -- pure function, safe for threading
# ---------------------------------------------------------------------------

def _call_api(origin: str, dest: str, date: str) -> list:
    """
    Single fast-flights call, nonstop only.
    Returns raw flight list or [] on any error.
    """
    try:
        result = get_flights(
            flight_data=[FlightData(date=date, from_airport=origin, to_airport=dest)],
            trip="one-way",
            seat="economy",
            passengers=Passengers(adults=1),
            max_stops=0,                    # nonstop flights only
        )
        return result.flights if result and result.flights else []
    except Exception:
        return []


def _departure_matches_date(departure_raw: str, date: str) -> bool:
    """
    Return True if departure_raw contains the search date (YYYY-MM-DD).
    Handles common formats like '2026-06-01 08:30', 'Mon Jun 1 8:30 AM', etc.
    Falls through to True when the date cannot be parsed, so we never
    silently drop rows whose format we don't recognise.
    """
    if not departure_raw:
        return True     # no timestamp to check -- let is_valid() handle it
    # Fast path: ISO date substring present
    if date in departure_raw:
        return True
    # Slower path: try to parse month/day from the search date and match
    try:
        target = datetime.strptime(date, "%Y-%m-%d")
        # Look for zero-padded or bare month-day patterns, e.g. "Jun 01" / "Jun 1"
        patterns = [
            target.strftime("%b %d"),   # "Jun 01"
            target.strftime("%b %-d"),  # "Jun 1"  (Linux/Mac only)
            target.strftime("%m/%d"),   # "06/01"
            target.strftime("%-m/%-d"), # "6/1"
        ]
        return any(p in departure_raw for p in patterns)
    except Exception:
        return True     # cannot determine -- keep the row


def fetch_pair(origin: str, dest: str, date: str, delay: float) -> list[dict]:
    """
    Call fast-flights for one (origin, dest) pair, nonstop only.
    If either airport is a hub and the first call returns nothing, retries up
    to MAX_RETRIES more times before giving up -- all within the same worker
    thread, no extra futures needed.
    Post-filters to:
      - nonstop only  (stops == 0, enforced both via API param and field check)
      - correct date  (departure timestamp matches the search date)
    Returns a list of validated, fully-populated row dicts.
    IDs are assigned here (after validation) so there are no gaps.
    """
    is_hub_pair = origin in HUB_AIRPORTS or dest in HUB_AIRPORTS
    attempts    = 1 + MAX_RETRIES if is_hub_pair else 1

    flights: list = []
    for attempt in range(attempts):
        flights = _call_api(origin, dest, date)
        if flights:
            break
        if attempt < attempts - 1:          # not the last attempt
            time.sleep(RETRY_BACKOFF)

    if delay > 0:
        time.sleep(delay)

    rows = []
    for flight in flights:
        # Double-check stops in case the API param is ignored / not supported
        if getattr(flight, "stops", 0) not in (0, None, ""):
            continue

        departure_raw = str(getattr(flight, "departure", "") or "").strip()

        # Skip flights whose departure timestamp belongs to a different date
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
    """Write validated flight rows + one search-log entry under one lock."""
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
    """
    Create output CSVs with headers if they don't exist.
    Returns the highest existing flight id so the counter resumes correctly
    when --resume is used.
    """
    fp = flight_csv_path(date)
    max_id = 0
    if not fp.exists():
        with open(fp, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FLIGHT_FIELDS).writeheader()
    else:
        with open(fp, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    max_id = max(max_id, int(row.get("id") or 0))
                except (ValueError, TypeError):
                    pass

    slp = search_log_path(date)
    if not slp.exists():
        with open(slp, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SEARCH_LOG_FIELDS).writeheader()

    return max_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape domestic US flight data using fast-flights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--date", required=True,
                        help="Departure date YYYY-MM-DD, e.g. 2026-06-01")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed pairs using the checkpoint file.")
    parser.add_argument("--continue", dest="continue_from_log", action="store_true",
                        help=(
                            "Skip pairs already recorded in searchLog_<date>.csv. "
                            "Use this after a completed run to scan only the new airports "
                            "added since then -- no checkpoint file needed."
                        ))
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent requests. Try 10-20; lower if errors spike.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Per-worker sleep (s) after each request. "
                             "Add ~0.5 if you hit rate limits.")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: --date must be YYYY-MM-DD, got: {args.date}")
        sys.exit(1)

    all_pairs   = list(itertools.permutations(AIRPORTS, 2))
    total_pairs = len(all_pairs)

    if args.continue_from_log:
        # Primary source: searchLog (works even if checkpoint was deleted)
        prior = load_completed_from_search_log(args.date)
        # Merge checkpoint too if it exists, so neither source is lost
        prior |= load_completed(args.date)
        print(f"Continuing from searchLog: {len(prior):,} pairs already done")
    elif args.resume:
        prior = load_completed(args.date)
    else:
        prior = set()
    _completed_shared.update(prior)

    pairs_to_run = [p for p in all_pairs if p not in _completed_shared]
    skipped      = total_pairs - len(pairs_to_run)

    max_id = ensure_csvs(args.date)
    init_id_counter(max_id)

    est_min = len(pairs_to_run) / max(args.workers, 1) * max(args.delay, 0.3) / 60
    print(f"Airports         : {len(AIRPORTS)}")
    print(f"Total pairs      : {total_pairs:,}")
    print(f"Already done     : {skipped:,}")
    print(f"Remaining        : {len(pairs_to_run):,}")
    print(f"Workers          : {args.workers}")
    print(f"Per-worker delay : {args.delay}s")
    print(f"Rough ETA        : ~{est_min:.0f} min")
    print(f"Flights CSV      : {flight_csv_path(args.date)}")
    print(f"Search-log CSV   : {search_log_path(args.date)}")
    print()

    total_flights = 0
    empty_pairs   = 0

    with (
        open(flight_csv_path(args.date), "a", newline="") as flight_file,
        open(search_log_path(args.date), "a", newline="") as log_file,
    ):
        flight_writer = csv.DictWriter(flight_file, fieldnames=FLIGHT_FIELDS)
        log_writer    = csv.DictWriter(log_file,    fieldnames=SEARCH_LOG_FIELDS)

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
                    total_flights += n
                    if n == 0:
                        empty_pairs += 1

                    pbar.set_postfix(flights=total_flights, empty=empty_pairs, refresh=False)
                    pbar.update(1)

    flush_checkpoint(args.date)

    print()
    print("Done.")
    print(f"  Valid flight rows  : {total_flights:,}  ->  {flight_csv_path(args.date)}")
    print(f"  Search-log entries : {len(pairs_to_run):,}  ->  {search_log_path(args.date)}")
    print(f"  Pairs with 0 valid flights : {empty_pairs:,} "
          f"({100 * empty_pairs / max(len(pairs_to_run), 1):.1f}%)")


if __name__ == "__main__":
    main()