from fast_flights import FlightData, Passengers, get_flights

result = get_flights(
    flight_data=[FlightData(date="2026-06-01", from_airport="IAD", to_airport="LAX")],
    trip="one-way",
    seat="economy",
    passengers=Passengers(adults=1),
)
print(result.flights)

# result.flights is a list of Flight objects with attributes:
# .name (airline), .departure, .arrival, .duration, .stops, .price, .is_best