# src/get_f1_session_data.py

import fastf1
import pandas as pd
import os

# Set up FastF1 cache
fastf1.Cache.enable_cache("data/cache")

def get_driver_laptimes(year, grand_prix, session_type="R"):
    """
    Load an F1 session and return driver numbers, names, teams, and average lap times.
    
    Args:
        year (int): The year of the race
        grand_prix (str): Name of the Grand Prix (e.g., "Monza")
        session_type (str): "R" for Race, "Q" for Qualifying
    
    Returns:
        pd.DataFrame: A DataFrame with DriverNumber, Driver, Team, AvgLapTime
    """
    print(f"Loading session data for {grand_prix} {year} ({session_type})...")
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()

    results = []

    for driver_code in session.drivers:
        laps = session.laps.pick_driver(driver_code).pick_quicklaps()
        if laps.empty:
            continue

        avg_lap = laps["LapTime"].mean().total_seconds()
        driver_info = session.get_driver(driver_code)

        results.append({
            "DriverNumber": driver_code,
            "Driver": driver_info["FullName"],
            "Team": driver_info["TeamName"],
            "AvgLapTime": round(avg_lap, 3)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = get_driver_laptimes(2023, "Monza", "R")

    os.makedirs("data/raw", exist_ok=True)
    output_file = "data/raw/monza_2023_race_laptimes.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved driver lap time data to: {output_file}")
    print(df.head())
