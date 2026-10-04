import sqlite3
conn = sqlite3.connect("flights.db")
conn.row_factory = sqlite3.Row
try:
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM flights;')
    total = cur.fetchone()[0]
    print(f'Flights table contains {total} entries.')
except sqlite3.OperationalError:
    print('Flights table does not exist.')
with conn:
    cur = conn.cursor()
    dest = input("Enter dest airport code: ").strip().upper()
    # cur.execute("SELECT id FROM flights WHERE dest = ?", (dest,))
    cur.execute('SELECT * FROM flights WHERE dest = ? LIMIT 10;', (dest,))
    rows = [r for r in cur.fetchall()]
    print(rows)
    for row in rows:
        r = dict(row)
        ns = f"SELECT * from flights WHERE expId = ?;"
        cur.execute(ns, (r['expId'],))
        newSql = [r for r in cur.fetchall()]
        print(r['expId'] + " - " + str(len(newSql)))
        # print(dict(row))