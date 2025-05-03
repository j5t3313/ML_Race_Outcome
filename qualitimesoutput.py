import fastf1
import pandas as pd

# 2025 target race 
year = 2025
gp   = "Miami"

#  load the 2024 session to get its grid
prev_session = fastf1.get_session(year-1, gp, 'Q')
prev_session.load()
# FastF1 puts the final qualifying classification into .results
grid_2024 = prev_session.results['Abbreviation'].unique()

# load the 2025 qualifying session
session = fastf1.get_session(year, gp, 'Q')
session.load()

laps = session.laps[['Driver','LapTime']].dropna()
best = (
    laps
    .groupby('Driver')['LapTime']
    .min()
    .reset_index()
)

# filter to only those who were on the 2024 grid
best = best[ best['Driver'].isin(grid_2024) ]

# Convert to seconds
best['QualifyingTime (s)'] = best['LapTime'].dt.total_seconds()

# Map codes back to full names
code_to_full = {
    "PIA": "Oscar Piastri",
    "RUS": "George Russell",
    "NOR": "Lando Norris",
    "VER": "Max Verstappen",
    "HAM": "Lewis Hamilton",
    "LEC": "Charles Leclerc",
    "TSU": "Yuki Tsunoda",
    "OCO": "Esteban Ocon",
    "HUL": "Nico Hülkenberg",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "SAI": "Carlos Sainz Jr.",
    "GAS": "Pierre Gasly",
    "BEA": "Ollie Bearman",
    "ANT": "Andrea Kimi Antonelli",
    "LAW": "Liam Lawson",
    "BOR": "Gabriel Bortoleto",
    "HAD": "Isack Hadjar",
    "DOO": "Jack Doohan",
    "ALB": "Alexander Albon"
}

# apply mapping & sort
best['Driver'] = best['Driver'].map(code_to_full)
best = best.sort_values('QualifyingTime (s)').reset_index(drop=True)

# build & print the final dict
output = {
    "Driver":             best['Driver'].tolist(),
    "QualifyingTime (s)": best['QualifyingTime (s)'].round(3).tolist()
}

print(output)
