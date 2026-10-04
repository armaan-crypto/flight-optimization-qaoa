-- legId, searchDate, flightDate, startingAirport, destinationAirport, totalFare, seatsRemaining, segmentsDepartureTimeRaw, segmentsArrivalTimeRaw, segmentsAirlineName, segmentsAirlineCode
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    expId TEXT NOT NULL,
    searchDate TEXT NOT NULL,
    origin TEXT NOT NULL,
    dest TEXT NOT NULL,
    totalFare TEXT NOT NULL,
    seatsRemaining TEXT NOT NULL,
    departure_time TEXT NOT NULL,
    arrival_time TEXT NOT NULL,
    airline TEXT NOT NULL,
    airlineCode TEXT NOT NULL
);