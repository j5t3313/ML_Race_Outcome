import fastf1
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# helper functions

def load_session(year, gp, session_types):
    """
    Try to load sessions in order; return (session, type_used).
    session_types: list like ['S', 'FP2']
    """
    for st in session_types:
        try:
            sess = fastf1.get_session(year, gp, st)
            sess.load()
            print(f"Loaded {st} session for {gp} {year}")
            return sess, st
        except Exception:
            continue
    raise RuntimeError(f"No session of types {session_types} found for {gp} {year}")


def get_pace_and_deg(session, min_laps=7):
    """
    From a loaded session, compute:
      - race_pace (mean lap) per driver
      - tire_deg (mean stint slope) per driver
      - stint_length stats and compound fractions
    """
    laps = session.laps
    # drop out-/in-laps
    laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna() & laps['LapTime'].notna()].copy()
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()

    # count laps per (Driver, Stint, Compound)
    laps['laps_in_group'] = (
        laps.groupby(['Driver','Stint','Compound'])['LapTime_s']
            .transform('count')
    )
    valid = laps[laps['laps_in_group'] >= min_laps].copy()

    # mean pace
    race_pace = valid.groupby('Driver')['LapTime_s'].mean()

    # tyre degradation: slope via direct Series.apply to avoid deprecation
    slopes = (
        valid.groupby(['Driver','Stint','Compound'])['LapTime_s']
             .apply(lambda t: np.polyfit(np.arange(len(t)), t.values, 1)[0] if len(t) > 1 else np.nan)
             .reset_index(name='slope')
    )
    tire_deg = slopes.groupby('Driver')['slope'].mean()

    # stint-length features
    stint_counts = (
        valid.groupby(['Driver','Stint'])
             .size()
             .reset_index(name='stint_laps')
    )
    stint_stats = stint_counts.groupby('Driver')['stint_laps'].agg(['max','median']).rename(
        columns={'max':'max_stint_laps','median':'median_stint_laps'}
    )

    # compound usage fractions
    comp_counts = valid.groupby(['Driver','Compound']).size().unstack(fill_value=0)
    comp_frac = comp_counts.div(comp_counts.sum(axis=1), axis=0)
    for c in ['Soft','Medium','Hard']:
        if c not in comp_frac.columns:
            comp_frac[c] = 0.0

    # assemble into df
    df = pd.concat([race_pace, tire_deg, stint_stats, comp_frac[['Soft','Medium','Hard']]], axis=1)
    df.columns = [
        'RacePace','TireDeg','MaxStintLaps','MedianStintLaps',
        'FracSoft','FracMedium','FracHard'
    ]
    # ensure index name matches merge key
    df.index.name = 'DriverCode'
    return df

# load & process 2024 data 
session_2024 = fastf1.get_session(2024, 'Monaco', 'R')
session_2024.load()
laps_2024 = session_2024.laps[['Driver','LapTime','Sector1Time','Sector2Time','Sector3Time']].dropna().copy()
for col in ['LapTime','Sector1Time','Sector2Time','Sector3Time']:
    laps_2024[f'{col} (s)'] = laps_2024[col].dt.total_seconds()

sector_times_2024 = (
    laps_2024.groupby('Driver')[['Sector1Time (s)','Sector2Time (s)','Sector3Time (s)']]
              .mean()
              .rename_axis('DriverCode')
              .reset_index()
)

# 2025 quali data
qualifying_2025 = pd.DataFrame({
    'Driver': ['Lando Norris', 'Charles Leclerc', 'Oscar Piastri', 'Lewis Hamilton', 'Max Verstappen', 'Alexander Albon', 'Fernando Alonso', 'Esteban Ocon', 'Carlos Sainz Jr.', 'Yuki Tsunoda', 'George Russell', 'Nico Hülkenberg', 'Pierre Gasly', 'Lance Stroll'
    ],
    'QualifyingTime (s)': [
        69.954, 70.063, 70.129, 70.382, 70.669, 70.732, 70.924, 70.942, 71.362, 71.415, 71.507, 71.596, 71.994, 72.563
    ]
})

driver_map = {
     "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Yuki Tsunoda": "TSU",
    "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL", "Fernando Alonso": "ALO", "Lance Stroll": "STR",
    "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS", "Andrea Kimi Antonelli": "ANT", "Ollie Bearman": "BEA", "Jack Doohan":"DOO", 
    "Gabriel Bortoleto":"BOR", "Isack Hadjar":"HAD", "Alexander Albon":"ALB", "Liam Lawson":"LAW", "Franco Colapinto":"COL"
}
qualifying_2025['DriverCode'] = qualifying_2025['Driver'].map(driver_map)

merged = qualifying_2025.merge(sector_times_2024, on='DriverCode', how='left')
unmapped = merged[merged[['Sector1Time (s)','Sector2Time (s)','Sector3Time (s)']].isna().any(axis=1)]
if not unmapped.empty:
    print('Unmapped drivers:', unmapped['Driver'].tolist())

# wet performance factor
wet_perf = {
    'Max Verstappen':0.975196,'Lewis Hamilton':0.976464,'Charles Leclerc':0.975862,
    'George Russell':0.968678,'Lando Norris':0.978179,'Yuki Tsunoda':0.996338,
    'Esteban Ocon':0.981810,'Fernando Alonso':0.972655,'Lance Stroll':0.979857,
    'Carlos Sainz Jr.':0.978754,'Pierre Gasly':0.978832,'Alexander Albon':0.978120
}
merged['WetPerformanceFactor'] = merged['Driver'].map(wet_perf).fillna(1.0)

# weather features 
API_KEY = 'APIKEY'
LAT, LON = '43.7402961','7.426559'
url = f'http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric'
wd = requests.get(url).json()

tz_local = pytz.timezone('Europe/Berlin')
race_local = tz_local.localize(datetime(2025,5,25,15,0,0))
race_utc = race_local.astimezone(pytz.UTC)

fd = min(wd['list'], key=lambda f: abs(pytz.UTC.localize(datetime.strptime(f['dt_txt'],'%Y-%m-%d %H:%M:%S')) - race_utc))
if abs(pytz.UTC.localize(datetime.strptime(fd['dt_txt'],'%Y-%m-%d %H:%M:%S')) - race_utc) < timedelta(hours=2):
    rain_prob = fd.get('pop',0)
    temp = fd['main']['temp']
    humidity = fd['main']['humidity']
    wind_speed = fd['wind']['speed']
    wind_dir = fd['wind']['deg']
else:
    rain_prob, temp, humidity, wind_speed, wind_dir = 0,22,50,0,0
for col,val in zip(
    ['RainProbability','Temperature','Humidity','WindSpeed','WindDir'],
    [rain_prob,temp,humidity,wind_speed,wind_dir]
): merged[col] = val

# pace & deg features
sess_p, stype = load_session(2025,'Monaco',['S','FP2'])
pace_deg_df = get_pace_and_deg(sess_p, min_laps=7)
pace_deg_df['SprintFlag'] = int(stype=='S')
pace_deg_df['TimeOfDay'] = race_local.hour
merged = merged.merge(pace_deg_df.reset_index(), on='DriverCode', how='left')

# derive an ordinal grid slot from QualifyingTime
# (1 = pole, 2 = P2, …)
merged['GridPos'] = merged['QualifyingTime (s)']\
    .rank(method='min')\
    .astype(int)

# constructor points feature
# map each driver to constructor
driver_to_constructor = {
    'VER': 'Red Bull Racing',
    'HAM': 'Ferrari',
    'LEC': 'Ferrari',
    'NOR': 'McLaren',
    'RUS': 'Mercedes',
    'ALO': 'Aston Martin',
    'OCO': 'Haas',
    'SAI': 'Williams',
    'TSU': 'Red Bull Racing',
    'HUL': 'Sauber',
    'GAS': 'Alpine',
    'STR': 'Aston Martin',
    'ALB': 'Williams',
    'PIA': 'McLaren',
    'LAW': 'Racing Bulls',
    'ANT': 'Mercedes',
    'BOR': 'Sauber',
    'HAD': 'Racing Bulls',
    'COL': 'Alpine',
    'BEA': 'Haas'
    
}

# hard‐code current championship points
constructor_points = {
    'Red Bull Racing': 143,
    'Mercedes':       147,
    'Ferrari':        142,
    'McLaren':        319,
    'Aston Martin':   14,
    'Alpine':         7,
    'Racing Bulls':      22,
    'Williams':        54,
    'Haas':            26,
    'Sauber': 6
}

merged['Constructor'] = merged['DriverCode'].map(driver_to_constructor)
merged['ConstructorPoints'] = merged['Constructor'].map(constructor_points)


features = [
    'QualifyingTime (s)','Sector1Time (s)','Sector2Time (s)','Sector3Time (s)',
    'WetPerformanceFactor','RainProbability','Temperature','Humidity','WindSpeed','WindDir',
    'RacePace','TireDeg','MaxStintLaps','MedianStintLaps',
    'FracSoft','FracMedium','FracHard','SprintFlag','TimeOfDay', 'ConstructorPoints', 'GridPos'
]
for feat in features:
    merged[f'{feat}_miss'] = merged[feat].isna().astype(int)
    merged[feat].fillna(merged[feat].median(), inplace=True)

# prepare X, y 
avg_lap = laps_2024.groupby('Driver')['LapTime (s)'].mean()
merged['AvgLapTime'] = merged['DriverCode'].map(avg_lap)
merged.dropna(subset=['AvgLapTime'], inplace=True)

X = merged[features + [f'{f}_miss' for f in features]]
y = merged['AvgLapTime']

# cross-validation & hyperparameter tuning 
base_model = GradientBoostingRegressor(random_state=38)
cv_scores = -cross_val_score(base_model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"CV MAE: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

param_dist = {
    'n_estimators': [100,200,300], 'learning_rate': [0.01,0.05,0.1],
    'max_depth': [3,5,7], 'subsample': [0.6,0.8,1.0]
}
rs = RandomizedSearchCV(
    base_model, param_dist, n_iter=10, cv=5,
    scoring='neg_mean_absolute_error', n_jobs=-1, random_state=38
)
rs.fit(X, y)
print("Best params:", rs.best_params_)

# final train/test split & evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)
model = rs.best_estimator_
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# final predictions & output 
valid_codes = merged['DriverCode'].unique()
qual = qualifying_2025[qualifying_2025['DriverCode'].isin(valid_codes)].copy()
qual['PredictedRaceTime (s)'] = model.predict(
    merged[merged['DriverCode'].isin(valid_codes)][features + [f'{f}_miss' for f in features]]
)
qual.sort_values('PredictedRaceTime (s)', inplace=True)
print("\n🏁 Predicted 2025 Monaco GP\n", qual[['Driver','PredictedRaceTime (s)']])
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.3f}, R²: {r2_score(y_test, y_pred):.3f}")

for name, imp in zip(features, model.feature_importances_):
    print(f"{name:20s} → {imp:.3f}")
