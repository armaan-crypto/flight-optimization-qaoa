import sqlite3
conn = sqlite3.connect('flights2.db')
conn.row_factory = sqlite3.Row

def create_table():
    sql = open("schema2.sql").read()
    cur = conn.cursor()
    cur.executescript(sql)
    conn.commit()

def insert(d):
    # return
    sql = '''INSERT INTO flights(expId, searchDate, origin, dest, totalFare, seatsRemaining, departure_time, arrival_time, airline, airlineCode)
             VALUES(?,?,?,?,?,?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, (d["expId"], d["searchDate"], d["startingAirport"], d["destinationAirport"], d["totalFare"], d["seatsRemaining"], d["segmentsDepartureTimeRaw"], d["segmentsArrivalTimeRaw"], d["segmentsAirlineName"], d["segmentsAirlineCode"]))
    conn.commit()
    return cur.lastrowid

# legId, searchDate, flightDate, startingAirport, destinationAirport, fareBasisCode, travelDuration, elapsedDays, isBasicEconomy, isRefundable, isNonStop, baseFare, totalFare, seatsRemaining, totalTravelDistance, segmentsDepartureTimeEpochSeconds, segmentsDepartureTimeRaw, segmentsArrivalTimeEpochSeconds, segmentsArrivalTimeRaw, segmentsArrivalAirportCode, segmentsDepartureAirportCode, segmentsAirlineName, segmentsAirlineCode, segmentsEquipmentDescription, segmentsDurationInSeconds, segmentsDistance, segmentsCabinCode
create_table()
with open("itineraries.csv") as f:
    f.readline()
    for line in f:
        data = line.strip().split(",")
        legId, searchDate, flightDate, startingAirport, destinationAirport, fareBasisCode, travelDuration, elapsedDays, isBasicEconomy, isRefundable, isNonStop, baseFare, totalFare, seatsRemaining, totalTravelDistance, segmentsDepartureTimeEpochSeconds, segmentsDepartureTimeRaw, segmentsArrivalTimeEpochSeconds, segmentsArrivalTimeRaw, segmentsArrivalAirportCode, segmentsDepartureAirportCode, segmentsAirlineName, segmentsAirlineCode, segmentsEquipmentDescription, segmentsDurationInSeconds, segmentsDistance, segmentsCabinCode = data
        if not (bool(isNonStop)): continue
        d = {
            "expId": legId,
            "searchDate": searchDate,
            "flightDate": flightDate,
            "startingAirport": startingAirport,
            "destinationAirport": destinationAirport,
            "totalFare": totalFare,
            "seatsRemaining": seatsRemaining,
            "segmentsDepartureTimeRaw": segmentsDepartureTimeRaw,
            "segmentsArrivalTimeRaw": segmentsArrivalTimeRaw,
            "segmentsAirlineName": segmentsAirlineName,
            "segmentsAirlineCode": segmentsAirlineCode
        }
        insert(d)
        # legId, searchDate, startingAirport, destinationAirport, totalFare, seatsRemaining, segmentsDepartureTimeRaw, segmentsArrivalTimeRaw, segmentsAirlineName, segmentsAirlineCode