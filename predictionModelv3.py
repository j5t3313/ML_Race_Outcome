import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib as plt

# Load previous year race session. 
session_2024 = fastf1.get_session(2024, "Miami", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 2025 Qualifying Data (Keeping Only 2024 Drivers)
qualifying_2025 = pd.DataFrame({
    'Driver': ['Max Verstappen', 'Lando Norris', 'Oscar Piastri', 'George Russell', 'Carlos Sainz Jr.', 'Alexander Albon', 'Charles Leclerc', 'Esteban Ocon', 'Yuki Tsunoda', 'Lewis Hamilton', 'Nico Hülkenberg', 'Fernando Alonso', 'Pierre Gasly', 'Lance Stroll'], 'QualifyingTime (s)': [86.204, 86.269, 86.269, 86.385, 86.569, 86.682, 86.754, 86.824, 86.943, 87.006, 87.473, 87.604, 87.71, 87.83]})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Yuki Tsunoda": "TSU",
    "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL", "Fernando Alonso": "ALO", "Lance Stroll": "STR",
    "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS", "Andrea Kimi Antonelli": "ANT", "Ollie Bearman": "BEA", "Jack Doohan":"DOO", 
    "Gabriel Bortoleto":"BOR", "Isack Hadjar":"HAD", "Alexander Albon":"ALB", "Liam Lawson":"LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

#
print (sector_times_2024)
print(qualifying_2025)

# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

#
print(merged_data)

#Driver-Specific Wet Performance based on 22-23 Canadian GP
driver_wet_performance = {
    "Max Verstappen": 0.975196, 
    "Lewis Hamilton": 0.976464, 
    "Charles Leclerc": 0.975862,
    "George Russell": 0.968678,
    "Lando Norris": 0.978179,
    "Yuki Tsunoda": 0.996338,
    "Esteban Ocon": 0.98181,
    "Fernando Alonso": 0.972655, 
    "Lance Stroll": 0.979857,
    "Carlos Sainz Jr.": 0.978754, 
    "Pierre Gasly": 0.978832,
    "Alexander Albon": 0.978120
    }

# Apply wet performance scores to qualifying times
#merged_data.rename(columns={"Driver_x": "Driver"}, inplace=True)
#print("Line 68 is running smoothly")
merged_data["WetPerformanceFactor"] = merged_data["Driver_x"].map(driver_wet_performance)
#
print(merged_data)

#Forecasted weather data for race using OpenWeatherMap API
API_KEY = "apikey" #Replace with your OpenWeatherMap API key
LAT = "25.7741728" # replace with latitude
LON = "-80.19362" # replace with longitude
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

# Fetch weather data
response = requests.get(weather_url)
weather_data = response.json()

#Extract the relevant weather data for the race (Sunday, 1400 hrs Local time)
forecast_time = "2025-05-04 16:00:00"
forecast_data = None
for forecast in weather_data["list"]:
    if forecast["dt_txt"] == forecast_time:
        forecast_data = forecast
        break

#Extract the weather features(Rain probability, temperature)
if forecast_data:
    rain_probability = forecast_data["pop"]  # Rain Probability (0-1 scale)
    temperature = forecast_data["main"]["temp"]  # Temperature in Celsius
else:
    rain_probability = 0 # Default if no data is found
    temperature = 28  # Assume Avg Temperature

# Create weather features for the model
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature


# Load Sprint if it exists otherwise FP2 2024   
try:
    sess_p = fastf1.get_session(2025, "Miami", "S")
    sess_p.load()
    print("Using Sprint session for race pace & deg.")
except Exception:
    sess_p = fastf1.get_session(2025, "Miami", "FP2")
    sess_p.load()
    print("Sprint not found, using FP2 session instead.")

# filter to only “normal” laps (no out-laps, no in-laps)  
lapsp = sess_p.laps[
    (sess_p.laps['PitInTime'].isna()) & 
    (sess_p.laps['PitOutTime'].isna()) & 
    sess_p.laps['LapTime'].notna()
].copy()

# convert times to seconds for convenience  
lapsp['LapTime_s'] = lapsp['LapTime'].dt.total_seconds()

# Race pace: mean lap time per driver
race_pace = lapsp.groupby('Driver')['LapTime_s'].mean()

# Tire degradation: average stint‐by‐stint slope  
def stint_slope(stint_df):
    t = stint_df['LapTime_s'].values
    if len(t) < 2:
        return np.nan
    # lap indices 0,1,2,... so slope = sec per lap
    return np.polyfit(np.arange(len(t)), t, 1)[0]

deg = (
    lapsp
    .groupby(['Driver','Stint'])
    .apply(stint_slope)
    .reset_index(name='slope')
    .groupby('Driver')['slope']
    .mean()
)

# Bring those back into merged_data via the 3-letter codes —  
merged_data['RacePace'] = merged_data['DriverCode'].map(race_pace)
merged_data['TireDeg']   = merged_data['DriverCode'].map(deg)

#Use Average Lap Time as target variable
# Compute avg lap time per code
avg_lap = laps_2024.groupby("Driver")["LapTime (s)"].mean()

# Map it back into merged_data
merged_data["AvgLapTime"] = merged_data["DriverCode"].map(avg_lap)

# Drop any rows where we failed to map (i.e. no target)
merged_data = merged_data.dropna(subset=["AvgLapTime"])

# Build X and y from the same DataFrame
X = merged_data[
    [
      "QualifyingTime (s)",
      "Sector1Time (s)",
      "Sector2Time (s)",
      "Sector3Time (s)",
      "WetPerformanceFactor",
      "RainProbability",
      "Temperature",
      "RacePace",
      "TireDeg"
    ]
].fillna(0)

y = merged_data["AvgLapTime"]

print(y)
print(X)


# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Figure out which codes actually made it into merged_data/X
valid_codes = merged_data["DriverCode"].unique().tolist()

# Filter your original qualifying_2025 to only those drivers so that it lines up 1:1 with the rows in X
qualifying_2025 = qualifying_2025[
    qualifying_2025["DriverCode"].isin(valid_codes)
].reset_index(drop=True)

# predict and assign:
qualifying_2025["PredictedRaceTime (s)"] = model.predict(X)

# sort, then print:
qualifying_2025 = (
    qualifying_2025
    .sort_values("PredictedRaceTime (s)")
    .reset_index(drop=True)
)
    
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])


# Print final predictions
print("\n🏁 Predicted 2025 Miami GP Winner (Excluding Rookies)🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")