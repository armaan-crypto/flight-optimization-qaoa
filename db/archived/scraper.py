"""
Domestic Nonstop Flight Scraper — fast-flights / Google Flights
================================================================
Key design decisions:
  1. Fetch ALL flights (no nonstop filter at request time) — fast-flights
     only returns Google's top ~5-10 picks. Filtering before fetch means
     you may get 0 results on busy routes like IAD->LAX where Google's
     top picks happen to all be connecting flights.
  2. Filter stops==0 AFTER fetching, from the full result set.
  3. Treat 0-row results on hub routes as suspicious and retry them,
     not just error rows — because fast-flights silently returns empty
     on many valid routes.
  4. Concurrent async with ThreadPoolExecutor (fast-flights is sync/blocking).

Install:
    pip install fast-flights pandas tqdm

Usage:
    python flight_scraper.py --date 2026-06-01
    python flight_scraper.py --date 2026-06-01 --concurrency 20
    python flight_scraper.py --date 2026-06-01 --tier1-only
    python flight_scraper.py --date 2026-06-01 --sample 100
    python flight_scraper.py --date 2026-06-01 --resume

Output:
    itineraries_2026-06-01.csv
    scrape_log_2026-06-01.csv
"""

import argparse
import asyncio
import csv
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from itertools import permutations
from pathlib import Path

import pandas as pd
from fast_flights import FlightData, Passengers, get_flights
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONCURRENCY  = 20
MAX_RETRIES  = 5
RETRY_PASSES = 3
BACKOFF_BASE = 2.0
BACKOFF_MAX  = 60.0

# Cycle fetch modes on each retry attempt
FETCH_MODES = ["fallback", "force-fallback", "common", "fallback", "force-fallback"]

_write_lock  = threading.Lock()
_seen_legids = set()


# ---------------------------------------------------------------------------
# Airport lists
# ---------------------------------------------------------------------------

TIER_1_HUBS = [
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "MIA", "IAH", "BOS", "MSP", "DTW", "FLL", "PHL",
    "LGA", "BWI", "SLC", "IAD", "DCA", "MDW", "HNL", "SAN", "TPA", "PDX",
    "STL", "BNA", "AUS", "MSY", "RDU", "DAL", "HOU", "OAK", "MCI", "SJC",
    "SMF", "SNA", "SJU", "OGG", "IND", "CVG", "PIT", "CMH", "SAT", "MEM",
]

TIER_2_AIRPORTS = [
    "ABQ", "ALB", "ANC", "BDL", "BHM", "BIL", "BOI", "BUF", "BUR",
    "CAE", "CAK", "CHS", "CID", "CLE", "COS", "CRP", "DAY", "DSM",
    "ELP", "FAR", "FAT", "FNT", "FSD", "GEG", "GPT", "GRR",
    "GSO", "GSP", "GTF", "HDN", "HSV", "ICT", "ILM", "JAC", "JAX",
    "JAN", "KOA", "LAN", "LBB", "LEX", "LFT", "LIT", "LNK",
    "MAF", "MFE", "MGM", "MKE", "MLB", "MLI", "MOB", "MRY", "MSN", "MTJ",
    "MYR", "OKC", "OMA", "ONT", "ORF", "PBI", "PNS", "PSC", "PSP",
    "PVD", "PWM", "RAP", "RDD", "RDM", "RIC", "RNO", "ROA", "ROC", "RST",
    "RSW", "SAV", "SBA", "SBN", "SBP", "SGF", "SHV", "SRQ",
    "SYR", "TLH", "TOL", "TUL", "TUS", "TYS", "VPS", "XNA", "YUM",
    "ACY", "AGS", "AMA", "AVL", "AVP", "AZO", "BFL", "BGR", "BTR",
    "BTV", "CHA", "CHO", "CMI", "DAB", "ERI", "EUG", "EYW", "FCA",
    "FLO", "FWA", "GFK", "GNV", "HPN", "HRL", "HTS", "IDA", "ITH",
    "LAW", "LCH", "LGB", "LSE", "MBS", "MEI", "MFR", "MHT",
    "MOT", "MSO", "OAJ", "OTH", "PHF", "PLN", "PRC", "PUB",
    "RHI", "SDF", "SPI", "SUX", "TRI", "TUP", "VLD",
]

ALL_AIRPORTS = list(dict.fromkeys(TIER_1_HUBS + TIER_2_AIRPORTS))


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_pairs(airports: list, sample: int = None) -> list:
    pairs = set()
    for o, d in permutations(TIER_1_HUBS, 2):
        pairs.add((o, d))
    non_hubs = [a for a in airports if a not in TIER_1_HUBS]
    for hub in TIER_1_HUBS:
        for airport in non_hubs:
            pairs.add((hub, airport))
            pairs.add((airport, hub))
    pairs = list(pairs)
    random.shuffle(pairs)
    return pairs[:sample] if sample else pairs


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def _fetch_all(origin: str, destination: str, flight_date: str,
               attempt_num: int) -> list:
    """
    Fetch ALL flights for this route (no stop filter at request time).
    Applies backoff if this is a retry attempt.
    Returns raw list of flight objects.
    """
    if attempt_num > 0:
        backoff = min(BACKOFF_BASE ** attempt_num + random.uniform(0, 1), BACKOFF_MAX)
        time.sleep(backoff)

    fetch_mode = FETCH_MODES[min(attempt_num, len(FETCH_MODES) - 1)]

    result = get_flights(
        flight_data=[FlightData(date=flight_date, from_airport=origin, to_airport=destination)],
        trip="one-way",
        seat="economy",
        passengers=Passengers(adults=1),
        fetch_mode=fetch_mode,
    )
    return result.flights if result and result.flights else []


def parse_duration(s) -> int:
    if not s:
        return None
    s = str(s)
    hrs = mins = 0
    if "hr" in s:
        parts = s.split("hr")
        try: hrs = int(parts[0].strip())
        except ValueError: pass
        s = parts[1] if len(parts) > 1 else ""
    if "min" in s:
        try: mins = int(s.replace("min", "").strip())
        except ValueError: pass
    total = hrs * 60 + mins
    return total or None


def parse_price(p) -> float:
    if p is None:
        return None
    try:
        return float(str(p).replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def flight_to_row(flight, origin, destination, search_date, flight_date) -> dict:
    try:
        elapsed = (datetime.strptime(flight_date, "%Y-%m-%d") -
                   datetime.strptime(search_date,  "%Y-%m-%d")).days
    except ValueError:
        elapsed = None

    airline  = str(getattr(flight, "name",      "") or "")
    dep      = str(getattr(flight, "departure",  "") or "")
    arr      = str(getattr(flight, "arrival",    "") or "")
    price    = parse_price(getattr(flight, "price",    None))
    duration = parse_duration(getattr(flight, "duration", None))
    leg_id   = f"{origin}-{destination}-{flight_date}-{dep}-{airline}".replace(" ", "_")

    return {
        "legId":                        leg_id,
        "searchDate":                   search_date,
        "flightDate":                   flight_date,
        "elapsedDays":                  elapsed,
        "startingAirport":              origin,
        "destinationAirport":           destination,
        "baseFare":                     price,
        "totalFare":                    price,
        "seatsRemaining":               None,
        "travelDuration":               duration,
        "isNonStop":                    True,
        "isBasicEconomy":               False,
        "isRefundable":                 None,
        "totalTravelDistance":          None,
        "segmentsDepartureAirportCode": origin,
        "segmentsArrivalAirportCode":   destination,
        "segmentsAirlineName":          airline,
        "segmentsAirlineCode":          "",
        "segmentsFlightNumber":         "",
        "segmentsDepartureTimeRaw":     f"{flight_date}T{dep}",
        "segmentsArrivalTimeRaw":       f"{flight_date}T{arr}",
        "segmentsCabinCode":            "COACH",
        "segmentsEquipmentCode":        "",
        "stops":                        0,
        "arrivalTimeAhead":             str(getattr(flight, "arrival_time_ahead", "") or ""),
        "isBestFlight":                 bool(getattr(flight, "is_best", False)),
        "delay":                        str(getattr(flight, "delay", "") or ""),
    }


def scrape_pair(origin, destination, flight_date, attempt_num) -> tuple:
    """
    Fetch all flights, filter to nonstop, convert to rows.
    Returns (rows, status, attempts).
    """
    try:
        all_flights = _fetch_all(origin, destination, flight_date, attempt_num)

        # ── Filter nonstop AFTER fetching the full result set ──────────────
        nonstop = [f for f in all_flights if (getattr(f, "stops", 1) or 0) == 0]

        rows = []
        for f in nonstop:
            try:
                rows.append(flight_to_row(
                    f, origin, destination,
                    date.today().strftime("%Y-%m-%d"), flight_date
                ))
            except Exception:
                continue

        # Distinguish: got flights but none nonstop vs truly empty
        if all_flights and not nonstop:
            status = "no_nonstop"   # flights exist, just no directs — don't retry
        elif rows:
            status = "ok"
        else:
            status = "empty"        # nothing returned at all — retry-eligible

        return rows, status, attempt_num + 1

    except Exception as e:
        return [], f"error: {str(e)[:80]}", attempt_num + 1


# ---------------------------------------------------------------------------
# CSV helpers (thread-safe, deduplicated)
# ---------------------------------------------------------------------------

ITINERARY_COLUMNS = [
    "legId", "searchDate", "flightDate", "elapsedDays",
    "startingAirport", "destinationAirport",
    "baseFare", "totalFare", "seatsRemaining",
    "travelDuration", "isNonStop", "isBasicEconomy", "isRefundable",
    "totalTravelDistance",
    "segmentsDepartureAirportCode", "segmentsArrivalAirportCode",
    "segmentsAirlineName", "segmentsAirlineCode", "segmentsFlightNumber",
    "segmentsDepartureTimeRaw", "segmentsArrivalTimeRaw",
    "segmentsCabinCode", "segmentsEquipmentCode",
    "stops", "arrivalTimeAhead", "isBestFlight", "delay",
]
LOG_COLUMNS = [
    "origin", "destination", "flightDate", "searchDate",
    "status", "rowsScraped", "attempts",
]


def init_csv(path, columns):
    if not Path(path).exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def safe_append_deduped(output_path, rows: list) -> int:
    global _seen_legids
    new_rows = []
    with _write_lock:
        for row in rows:
            lid = row["legId"]
            if lid not in _seen_legids:
                _seen_legids.add(lid)
                new_rows.append(row)
        if new_rows:
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=ITINERARY_COLUMNS,
                               extrasaction="ignore").writerows(new_rows)
    return len(new_rows)


def safe_append_log(log_path, row: dict):
    with _write_lock:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS,
                           extrasaction="ignore").writerow(row)


def load_seen_legids(output_path):
    global _seen_legids
    if not Path(output_path).exists():
        return
    try:
        df = pd.read_csv(output_path, usecols=["legId"])
        _seen_legids.update(df["legId"].dropna().tolist())
        print(f"  Loaded {len(_seen_legids):,} existing legIds for deduplication.")
    except Exception:
        pass


def get_status_map(log_path) -> dict:
    """Returns {(origin, dest): (status, attempts)} for all logged pairs."""
    result = {}
    if not Path(log_path).exists():
        return result
    try:
        df = pd.read_csv(log_path)
        for _, r in df.iterrows():
            pair = (r["origin"], r["destination"])
            result[pair] = (str(r.get("status", "")), int(r.get("attempts", 1)))
    except Exception:
        pass
    return result


def should_retry(status: str, attempts: int) -> bool:
    """
    Retry if:
      - status is an error (network/auth issue)
      - status is 'empty' (fast-flights returned nothing — suspicious on major routes)
    Do NOT retry:
      - 'ok' (success)
      - 'no_nonstop' (flights exist, just no directs — legitimate)
      - exhausted retries
    """
    if attempts >= MAX_RETRIES:
        return False
    return status.startswith("error") or status == "empty"


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def process_pair(executor, semaphore, origin, dest, flight_date,
                       output_file, log_file, pbar, counter, attempt_num=0):
    async with semaphore:
        loop = asyncio.get_event_loop()
        rows, status, total_attempts = await loop.run_in_executor(
            executor, scrape_pair, origin, dest, flight_date, attempt_num
        )

    written = safe_append_deduped(output_file, rows) if rows else 0
    
    safe_append_log(log_file, {
        "origin": origin, "destination": dest,
        "flightDate": flight_date,
        "searchDate": date.today().strftime("%Y-%m-%d"),
        "status": status, "rowsScraped": written, "attempts": total_attempts,
    })

    with _write_lock:
        counter[0] += written

    if pbar:
        pbar.set_postfix({"rows": counter[0], "route": f"{origin}->{dest}",
                          "status": status[:12]})
        pbar.update(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    flight_date = args.date
    output_file = f"itineraries_{flight_date}.csv"
    log_file    = f"scrape_log_{flight_date}.csv"

    print("=" * 60)
    print(f"  Nonstop Flight Scraper  —  {flight_date}")
    print("=" * 60)

    airports    = TIER_1_HUBS if args.tier1_only else ALL_AIRPORTS
    route_pairs = generate_pairs(airports, sample=args.sample)

    print(f"Route pairs:  {len(route_pairs):,}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Retry passes: {RETRY_PASSES}  (retries errors + suspicious empties)")
    print(f"Filter:       nonstop only (applied post-fetch)\n")

    init_csv(output_file, ITINERARY_COLUMNS)
    init_csv(log_file,    LOG_COLUMNS)

    if args.resume:
        load_seen_legids(output_file)

    counter = [0]

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:

        # ── Pass 1: main scrape ──────────────────────────────────────────
        status_map = get_status_map(log_file) if args.resume else {}
        # Skip pairs that already succeeded or confirmed no nonstop service
        tasks = [(o, d) for (o, d) in route_pairs
                 if status_map.get((o, d), ("", 0))[0] not in ("ok", "no_nonstop")]
        random.shuffle(tasks)
        print(f"── Pass 1: {len(tasks):,} pairs ──")

        sem = asyncio.Semaphore(args.concurrency)
        with tqdm(total=len(tasks), desc="Pass 1", unit="pair", dynamic_ncols=True) as pbar:
            await asyncio.gather(*(
                process_pair(executor, sem, o, d, flight_date,
                             output_file, log_file, pbar, counter, attempt_num=0)
                for (o, d) in tasks
            ))

        # ── Retry passes ─────────────────────────────────────────────────
        for retry_pass in range(1, RETRY_PASSES + 1):
            status_map = get_status_map(log_file)
            to_retry = [
                (o, d, attempts)
                for (o, d), (status, attempts) in status_map.items()
                if should_retry(status, attempts)
            ]

            if not to_retry:
                print(f"\nNo retryable pairs remaining. Done!")
                break

            print(f"\n── Retry pass {retry_pass}: {len(to_retry):,} pairs ──")
            retry_sem = asyncio.Semaphore(max(5, args.concurrency // 2))

            with tqdm(total=len(to_retry), desc=f"Retry {retry_pass}",
                      unit="pair", dynamic_ncols=True) as pbar:
                await asyncio.gather(*(
                    process_pair(executor, retry_sem, o, d, flight_date,
                                 output_file, log_file, pbar, counter, attempt_num=attempts)
                    for (o, d, attempts) in to_retry
                ))

    # ── Summary ──────────────────────────────────────────────────────────
    status_map   = get_status_map(log_file)
    still_bad    = [(o, d) for (o, d), (st, at) in status_map.items()
                    if should_retry(st, at)]
    no_nonstop   = [(o, d) for (o, d), (st, _) in status_map.items()
                    if st == "no_nonstop"]

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Nonstop rows written:       {counter[0]:,}")
    print(f"  Routes with no nonstop:     {len(no_nonstop):,}  (legitimate — no direct service)")
    print(f"  Pairs still unresolved:     {len(still_bad):,}  (see {log_file})")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape nonstop domestic US flights for a specific departure date."
    )
    parser.add_argument("--date", type=str, required=True,
        help="Departure date (YYYY-MM-DD), e.g. --date 2026-06-01")
    parser.add_argument("--sample", type=int, default=None,
        help="Sample N route pairs (for testing).")
    parser.add_argument("--tier1-only", action="store_true",
        help="Only scrape the 50 Tier-1 hub airports.")
    parser.add_argument("--resume", action="store_true",
        help="Skip pairs already marked ok/no_nonstop in the log.")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
        help=f"Simultaneous requests (default: {CONCURRENCY}).")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print("Error: --date must be YYYY-MM-DD.")
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()