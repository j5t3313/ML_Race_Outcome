import fastf1
import pandas as pd

#Load the 2023 Canadian Grand Prix(Wet Race)
session_2023 = fastf1.get_session(2023, "Canada", "R")
session_2023.load()

#Load the 2022 Canadian Grand Prix(Dry race)
session_2022 = fastf1.get_session(2022, "Canada", "R")
session_2022.load()

"""#check
print(type(session_2023.laps))            # Should be <class 'pandas.core.frame.DataFrame'>
print(session_2023.laps.head())           # Show first few rows
print(session_2023.laps.columns.tolist()) # Show column names"""

# Extract Lap times from both sessions
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()
laps_2022 = session_2022.laps[["Driver", "LapTime"]].copy()

#Drop NaN values
laps_2023.dropna(inplace=True)
laps_2022.dropna(inplace=True)

# Convert Lap times to seconds
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()
laps_2022["LapTime (s)"] = laps_2022["LapTime"].dt.total_seconds()

#Calculate the average lap time for each driver in both races
avg_lap_times_2023 = laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_lap_times_2022 = laps_2022.groupby("Driver")["LapTime (s)"].mean().reset_index()

#Merge the data from both races on drivers' column
merged_data = pd.merge(avg_lap_times_2023, avg_lap_times_2022, on="Driver", suffixes=("_2023", "_2022"))

# Calculate the difference in lap times between 2022 and 2023
merged_data["LapTimeDifference (s)"] = merged_data["LapTime (s)_2023"] - merged_data["LapTime (s)_2022"]

#Calculate the percentage change in lap times between 2022 and 2023
merged_data["LapTimePercentageChange (%)"] = (merged_data["LapTimeDifference (s)"] / merged_data["LapTime (s)_2022"]) * 100

#Creating a wet performance score
merged_data["WetPerformanceScore"] = 1+ (merged_data["LapTimePercentageChange (%)"] / 100)
print(merged_data[["Driver", "WetPerformanceScore"]])