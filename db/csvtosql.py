# import_nonstop_csv_to_sqlite.py

import sqlite3
import csv

CSV_FILE = "final.csv"
DB_FILE = "nonstop.db"
TABLE_NAME = "flights"

# Connect to SQLite database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Read CSV headers
with open(CSV_FILE, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)

# Create table with all columns as TEXT except numeric ones
column_types = {
    "id": "INTEGER",
    "travelDuration": "INTEGER",
    "totalFare": "REAL",
    "isBestFlight": "BOOLEAN",
}

columns_sql = []
for header in headers:
    col_type = column_types.get(header, "TEXT")
    if header == "id":
        columns_sql.append(f'"{header}" {col_type} PRIMARY KEY')
    else:
        columns_sql.append(f'"{header}" {col_type}')

create_table_sql = f'''
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    {", ".join(columns_sql)}
)
'''
print(create_table_sql)
input()

cursor.execute(create_table_sql)

# Insert CSV rows
placeholders = ", ".join(["?"] * len(headers))
insert_sql = f'''
INSERT OR REPLACE INTO {TABLE_NAME}
({", ".join([f'"{h}"' for h in headers])})
VALUES ({placeholders})
'''

with open(CSV_FILE, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    rows = []
    for row in reader:
        # Convert types
        if row["totalFare"] == "": continue
        row["id"] = int(row["id"])
        td = row.get("travelDuration", "").strip()
        if ":" in td:
            try:
                h, m = td.split(":", 1)
                row["travelDuration"] = int(h) * 60 + int(m)
            except ValueError:
                row["travelDuration"] = 0
        row["travelDuration"] = int(row["travelDuration"])
        row["totalFare"] = float(row["totalFare"])
        row["isBestFlight"] = 1 if row["isBestFlight"] == "True" else 0

        rows.append(tuple(row[h] for h in headers))

    cursor.executemany(insert_sql, rows)

# Create indexes for fast searching
cursor.execute(f'''
CREATE INDEX IF NOT EXISTS idx_departure
ON {TABLE_NAME}(segmentsDepartureAirportCode)
''')

cursor.execute(f'''
CREATE INDEX IF NOT EXISTS idx_arrival
ON {TABLE_NAME}(segmentsArrivalAirportCode)
''')

# Optional combined index for common route searches
cursor.execute(f'''
CREATE INDEX IF NOT EXISTS idx_route
ON {TABLE_NAME}(
    segmentsDepartureAirportCode,
    segmentsArrivalAirportCode
)
''')

conn.commit()
conn.close()

print(f"Imported {CSV_FILE} into {DB_FILE}")