#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

INPUT_FILES = ["final.csv", "removedduplicates.csv"]
OUTPUT_FILE = "finalfinal.csv"
FIELDS = [
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

def make_key(row):
    parts = []
    for f in FIELDS:
        if f == "id":
            continue
        v = (row.get(f) or "").strip()
        if f == "segmentsAirlineName":
            v = v.lower()
        if f.endswith("AirportCode") or f == "segmentsCabinCode":
            v = v.upper()
        parts.append(v)
    return "".join(parts)

def main():
    seen = set()
    out_rows = []
    next_id = 1

    for fname in INPUT_FILES:
        p = Path(fname)
        if not p.exists():
            print(f"warning: {fname} not found, skipping", file=sys.stderr)
            continue
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = make_key(r)
                if key in seen:
                    continue
                seen.add(key)
                row = {k: r.get(k, "") for k in FIELDS}
                row["id"] = str(next_id)
                next_id += 1
                out_rows.append(row)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

if __name__ == "__main__":
    main()