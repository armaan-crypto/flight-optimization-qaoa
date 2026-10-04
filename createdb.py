import sqlite3
conn = sqlite3.connect('flights.db')
conn.row_factory = sqlite3.Row

def create_table():
    sql = open("schema.sql").read()
    cur = conn.cursor()
    cur.executescript(sql)
    conn.commit()

def insert(d):
    sql = '''INSERT INTO flights(year, month, day, airline, flight_number, origin, dest, departure_time, arrival_time)
             VALUES(?,?,?,?,?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, (d["year"], d["month"], d["day"], d["airline"], d["flight_number"], d["origin_airport"], d["destination_airport"], d["scheduled_departure"], d["scheduled_arrival"]))
    conn.commit()
    return cur.lastrowid

cursor = conn.cursor()
# YEAR,MONTH,DAY,DAY_OF_WEEK,AIRLINE,FLIGHT_NUMBER,TAIL_NUMBER,ORIGIN_AIRPORT,DESTINATION_AIRPORT,SCHEDULED_DEPARTURE,DEPARTURE_TIME,DEPARTURE_DELAY,TAXI_OUT,WHEELS_OFF,SCHEDULED_TIME,ELAPSED_TIME,AIR_TIME,DISTANCE,WHEELS_ON,TAXI_IN,SCHEDULED_ARRIVAL,ARRIVAL_TIME,ARRIVAL_DELAY,DIVERTED,CANCELLED,CANCELLATION_REASON,AIR_SYSTEM_DELAY,SECURITY_DELAY,AIRLINE_DELAY,LATE_AIRCRAFT_DELAY,WEATHER_DELAY
flights = []
create_table()
with open("flights.csv") as f:
    f.readline()
    day1 = -1
    for line in f:
        data = line.strip().split(",")
        year = int(data[0])
        month = int(data[1])
        day = int(data[2])
        airline = data[4]
        flight_number = data[5]
        origin_airport = data[7]
        destination_airport = data[8]
        scheduled_departure = data[9]
        scheduled_arrival = data[20]
        if day1 == -1: day1 = day
        if day != day1: break
        d = {
            "year": year,
            "month": month,
            "day": day,
            "airline": airline,
            "flight_number": flight_number,
            "origin_airport": origin_airport,
            "destination_airport": destination_airport,
            "scheduled_departure": scheduled_departure,
            "scheduled_arrival": scheduled_arrival
        }
        insert(d)
        flights.append(d)
print(len(flights))