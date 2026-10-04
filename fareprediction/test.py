from fare_evolution import project_fare

price = project_fare(
    snapshot_price=118.0,
    snapshot_days_out=20,
    target_days_out=19,
    airline="Delta",
    cabin="COACH",
    is_nonstop=True,
    seats_remaining=55,
)
print(price_in_2_days)