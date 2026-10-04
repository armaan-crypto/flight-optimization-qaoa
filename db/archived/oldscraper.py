"""
Domestic Flight Scraper — Google Flights via fast-flights (CONCURRENT + AUTO-RETRY)
=====================================================================================
Automatically re-attempts any failed route pairs after the main pass completes,
with exponential backoff and fallback fetch modes to handle:
  - 401 "no token provided" (Google auth challenge — retry with force-fallback)
  - Network/connection errors (transient — retry after backoff)
  - Empty results (legitimate no-service routes — logged, not retried)

Install:
    pip install fast-flights pandas tqdm

Usage:
    python flight_scraper.py --date 2026-06-01
    python flight_scraper.py --date 2026-06-01 --concurrency 30
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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from itertools import permutations
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from fast_flights import FlightData, Passengers, get_flights


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONCURRENCY    = 20
MAX_RETRIES    = 5          # per pair across all passes
RETRY_PASSES   = 3          # how many retry rounds after the main pass
BACKOFF_BASE   = 2.0        # exponential backoff base (seconds)
BACKOFF_MAX    = 60.0       # cap on backoff wait

_write_lock = threading.Lock()

# Fetch modes to cycle through on successive retries
FETCH_MODES = ["fallback", "force-fallback", "common", "fallback", "force-fallback"]


# ---------------------------------------------------------------------------
# US Domestic Airport Lists
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
    "BTV", "CHA", "CHO", "CMI", "DAB", "ERI",
    "EUG", "EYW", "FCA", "FLO", "FWA", "GFK", "GNV",
    "HPN", "HRL", "HTS", "IDA", "ITH", "LAW", "LCH", "LGB",
    "LSE", "MBS", "MEI", "MFR", "MHT",
    "MOT", "MSO", "OAJ", "OTH", "PHF",
    "PLN", "PRC", "PUB", "RHI", "SDF", "SPI",
    "SUX", "TRI", "TUP", "VLD",
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
# Data extraction
# ---------------------------------------------------------------------------

def _blocking_fetch(origin: str, destination: str, flight_date: str,
                    fetch_mode: str = "fallback") -> list:
    """Single attempt — raises on failure so caller can handle retry logic."""
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
    stops    = int(getattr(flight, "stops",      0)  or 0)
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
        "isNonStop":                    stops == 0,
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
        "stops":                        stops,
        "arrivalTimeAhead":             str(getattr(flight, "arrival_time_ahead", "") or ""),
        "isBestFlight":                 bool(getattr(flight, "is_best", False)),
        "delay":                        str(getattr(flight, "delay", "") or ""),
    }


# ---------------------------------------------------------------------------
# CSV helpers (thread-safe)
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


def safe_append(path, columns, rows):
    with _write_lock:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns, extrasaction="ignore").writerows(rows)


def update_log_status(log_path, origin, dest, flight_date, new_status, rows_scraped, attempts):
    """Overwrite the status for an existing (origin, dest) pair in the log."""
    with _write_lock:
        try:
            df = pd.read_csv(log_path)
            mask = (df["origin"] == origin) & (df["destination"] == dest) & \
                   (df["flightDate"] == flight_date)
            if mask.any():
                df.loc[mask, "status"]      = new_status
                df.loc[mask, "rowsScraped"] = rows_scraped
                df.loc[mask, "attempts"]    = attempts
                df.to_csv(log_path, index=False)
                return
        except Exception:
            pass
    # If not found, append new row
    safe_append(log_path, LOG_COLUMNS, [{
        "origin": origin, "destination": dest, "flightDate": flight_date,
        "searchDate": date.today().strftime("%Y-%m-%d"),
        "status": new_status, "rowsScraped": rows_scraped, "attempts": attempts,
    }])


def load_completed(log_path) -> set:
    """Return (origin, dest) pairs with status 'ok' or 'empty' (no retry needed)."""
    done = set()
    if not Path(log_path).exists():
        return done
    try:
        df = pd.read_csv(log_path)
        for _, r in df.iterrows():
            if r["status"] in ("ok", "empty"):
                done.add((r["origin"], r["destination"]))
    except Exception:
        pass
    return done


def load_failed(log_path) -> list:
    """Return (origin, dest, attempts) for rows that still have error status."""
    failed = []
    if not Path(log_path).exists():
        return failed
    try:
        df = pd.read_csv(log_path)
        for _, r in df.iterrows():
            if str(r["status"]).startswith("error"):
                attempts = int(r.get("attempts", 1))
                failed.append((r["origin"], r["destination"], attempts))
    except Exception:
        pass
    return failed


# ---------------------------------------------------------------------------
# Core fetch with retry + backoff (runs inside a thread)
# ---------------------------------------------------------------------------

def fetch_with_retry(origin: str, destination: str, flight_date: str,
                     attempt_num: int) -> tuple:
    """
    Try fetching with escalating fetch modes and backoff.
    Returns (flights, final_status, total_attempts).
    attempt_num: which attempt this is (0-indexed), used to pick fetch mode + backoff.
    """
    fetch_mode = FETCH_MODES[min(attempt_num, len(FETCH_MODES) - 1)]
    backoff    = min(BACKOFF_BASE ** attempt_num + random.uniform(0, 1), BACKOFF_MAX)

    if attempt_num > 0:
        time.sleep(backoff)

    try:
        flights = _blocking_fetch(origin, destination, flight_date, fetch_mode)
        status  = "ok" if flights else "empty"
        return flights, status, attempt_num + 1
    except Exception as e:
        return [], f"error: {str(e)[:80]}", attempt_num + 1


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def process_pair(executor, semaphore, origin, dest, flight_date, search_date,
                       output_file, log_file, pbar, counter, attempt_num=0):
    """
    Process one (origin, dest) pair. Returns True if succeeded, False if failed.
    """
    rows   = []
    status = "ok"
    total_attempts = attempt_num

    async with semaphore:
        loop = asyncio.get_event_loop()
        try:
            flights, status, total_attempts = await loop.run_in_executor(
                executor, fetch_with_retry, origin, dest, flight_date, attempt_num
            )
            for f in flights:
                try:
                    rows.append(flight_to_row(f, origin, dest, search_date, flight_date))
                except Exception:
                    continue
        except Exception as e:
            status = f"error: {str(e)[:80]}"
            total_attempts = attempt_num + 1

    if rows:
        safe_append(output_file, ITINERARY_COLUMNS, rows)

    with _write_lock:
        counter[0] += len(rows)

    if pbar:
        pbar.set_postfix({"rows": counter[0], "route": f"{origin}->{dest}", "status": status[:20]})
        pbar.update(1)

    # Write / update log
    update_log_status(log_file, origin, dest, flight_date, status, len(rows), total_attempts)

    return not status.startswith("error")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    flight_date = args.date
    search_date = date.today().strftime("%Y-%m-%d")
    output_file = f"itineraries_{flight_date}.csv"
    log_file    = f"scrape_log_{flight_date}.csv"

    print("=" * 60)
    print(f"  Domestic Flight Scraper  —  departures on {flight_date}")
    print("=" * 60)

    airports    = TIER_1_HUBS if args.tier1_only else ALL_AIRPORTS
    route_pairs = generate_pairs(airports, sample=args.sample)

    print(f"Route pairs:  {len(route_pairs):,}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Retry passes: {RETRY_PASSES}")

    init_csv(output_file, ITINERARY_COLUMNS)
    init_csv(log_file,    LOG_COLUMNS)

    # --- PASS 1: Main scrape ---
    completed = load_completed(log_file) if args.resume else set()
    if completed:
        print(f"\nResuming — {len(completed):,} pairs already done.")

    tasks = [(o, d) for (o, d) in route_pairs if (o, d) not in completed]
    random.shuffle(tasks)
    print(f"Pairs remaining: {len(tasks):,}\n")

    counter   = [0]
    semaphore = asyncio.Semaphore(args.concurrency)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        print("── Pass 1: Main scrape ──")
        with tqdm(total=len(tasks), desc="Pass 1", unit="pair", dynamic_ncols=True) as pbar:
            await asyncio.gather(*(
                process_pair(executor, semaphore, o, d, flight_date, search_date,
                             output_file, log_file, pbar, counter, attempt_num=0)
                for (o, d) in tasks
            ))

        # --- RETRY PASSES ---
        for retry_pass in range(1, RETRY_PASSES + 1):
            failed = load_failed(log_file)
            # Only retry pairs that haven't exceeded MAX_RETRIES
            to_retry = [(o, d, attempts) for (o, d, attempts) in failed
                        if attempts < MAX_RETRIES]

            if not to_retry:
                print(f"\nNo failures remaining after pass {retry_pass - 1}. Done!")
                break

            print(f"\n── Pass {retry_pass + 1}: Retrying {len(to_retry):,} failed pairs ──")
            # Slow down retries slightly — reduce concurrency to ease auth pressure
            retry_concurrency = max(5, args.concurrency // 2)
            retry_semaphore   = asyncio.Semaphore(retry_concurrency)

            with tqdm(total=len(to_retry), desc=f"Retry {retry_pass}",
                      unit="pair", dynamic_ncols=True) as pbar:
                await asyncio.gather(*(
                    process_pair(executor, retry_semaphore, o, d, flight_date, search_date,
                                 output_file, log_file, pbar, counter, attempt_num=attempts)
                    for (o, d, attempts) in to_retry
                ))

    # --- Final summary ---
    final_failed = load_failed(log_file)
    exhausted    = [(o, d, a) for (o, d, a) in final_failed if a >= MAX_RETRIES]

    print(f"\n{'='*60}")
    print(f"  Complete!")
    print(f"  Rows written:       {counter[0]:,}")
    print(f"  Output file:        {output_file}")
    print(f"  Pairs still failed: {len(exhausted):,}  (check {log_file})")
    print(f"{'='*60}")

    if exhausted:
        print(f"\nPersistently failed pairs (no flights found after {MAX_RETRIES} attempts):")
        for o, d, _ in exhausted[:20]:
            print(f"  {o} -> {d}")
        if len(exhausted) > 20:
            print(f"  ... and {len(exhausted) - 20} more (see log file)")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape all domestic US flights departing on a specific date, with auto-retry."
    )
    parser.add_argument("--date", type=str, required=True,
        help="Target departure date (YYYY-MM-DD), e.g. --date 2026-06-01")
    parser.add_argument("--sample", type=int, default=None,
        help="Randomly sample N route pairs (for testing).")
    parser.add_argument("--tier1-only", action="store_true",
        help="Only scrape the 50 Tier-1 hub airports.")
    parser.add_argument("--resume", action="store_true",
        help="Skip pairs already marked ok/empty in the log.")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
        help=f"Simultaneous requests (default: {CONCURRENCY}).")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print("Error: --date must be in YYYY-MM-DD format.")
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()