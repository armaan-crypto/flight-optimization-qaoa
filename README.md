# Flight Optimization using the Quantum Approximate Optimization Algorithm (QAOA)

Finding the best multi-hop itinerary between two US airports, framed as a combinatorial optimization problem and solved with a QAOA-based optimizer. A classical A\* search runs on the same data as a baseline, so the two approaches can be compared on route cost, optimality gap, feasibility and runtime.

## Overview

Choosing a flight route (which connections to take, at what times, on which carriers) is a constrained shortest-path problem. This project:

1. **Collects** real fare data for domestic US flights (scraped with `fast-flights`).
2. **Stores** it in SQLite (`flights.db` for the full dataset, `db/nonstop.db` for the cleaned nonstop subset).
3. **Projects** fares across booking horizons with a fare-evolution model (`db/fare_evolution.py`).
4. **Optimizes** routes with a QAOA-style optimizer (`db/qaoa.py`) and with A\* (`db/astar2.py`).
5. **Benchmarks** both solvers across network sizes and produces figures (`db/testing.py`, `db/analyze.py`).

## Route scoring

Each candidate route gets a score in (0, 1), lower is better. Weighted components are normalized, summed, then passed through a sigmoid:

| Component | Weight | Meaning |
|---|---|---|
| Predicted fare | 3.0 | Dominant term |
| Layover | 1.0 | Deviation from the ideal layover (2 h by default) |
| Segments | 0.8 | Fewer stops preferred |
| Airline | 0.6 | Preferred-carrier bonus |
| Departure time | 0.3 | Prefer mid-window departures |
| Leg travel time | 0.3 | Prefer shorter legs |

Search-space pruning keeps the problem tractable: a distance-based cap on segments (up to 4), a total-distance cap relative to the direct distance, layover limits (1–6 h), hub-rank pruning, geographic detour pruning, and optional airline/airport filters, departure/arrival windows and a total travel-time cap. All of these are set in the `CONFIG` block at the top of `db/qaoa.py`.

## Repository layout

```
.
├── createdb.py / schema2.sql      Create the SQLite flights table
├── scraper.py                     Early Expedia/Selenium scraper
├── astar.py, astar2.py            Early A* route search prototypes
├── cheapest_flight_path.py        Cheapest-path search against flights.db
├── db/
│   ├── flight_scraper.py          Domestic flight scraper (fast-flights)
│   ├── csvtosql.py                final.csv -> nonstop.db
│   ├── qaoa.py                    QAOA flight optimizer (edit CONFIG, then run)
│   ├── astar2.py                  A* baseline
│   ├── fare_evolution.py          Fare projection vs. days-before-departure
│   ├── testing.py                 Benchmark harness -> results.csv
│   ├── analyze.py                 Figures from results.csv
│   ├── figures/, figures2/        Generated plots
│   ├── nonstop.db, final.csv      Cleaned nonstop dataset
│   └── results.csv, results2.csv  Benchmark output
├── fareprediction/                Fare model experiments
└── origins.txt                    Origin airports in the dataset
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r db/req.txt

cd db
# Edit ORIGIN / DESTINATION and constraints in the CONFIG block of qaoa.py
python qaoa.py
```

Run the benchmark and generate figures:

```bash
python testing.py      # writes results.csv
python analyze.py      # writes figures
```

## Data

`flights.db` (~35 GB of scraped fares) is **not** included in this repository because of its size and is listed in `.gitignore`. To work with the full dataset, rebuild it using `createdb.py` and `db/flight_scraper.py`. The smaller `db/nonstop.db` is included and is enough to run the optimizer.

## Results

Benchmark output is in `db/results.csv` and the plots in `db/figures/`: route cost, optimality gap, feasibility, runtime scaling, nodes expanded vs. iterations, and fare-projection error.

## Author

Armaan Ahmed
