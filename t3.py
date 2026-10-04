import sqlite3
conn = sqlite3.connect("flights.db")
conn.row_factory = sqlite3.Row

cur = conn.cursor()
cur.execute("SELECT DISTINCT origin FROM flights")
origins = {row["origin"] for row in cur.fetchall() if row["origin"] is not None}
print(origins)
with open("origins.txt", "w") as f:
    for origin in origins:
        f.write(origin + "\n")
cur.close()
conn.close()