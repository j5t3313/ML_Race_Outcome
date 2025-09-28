import fastf1
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RaceConfig:
    year: int = 2025
    gp_name: str = 'Canadian GP'
    min_stint_laps: int = 7
    api_key: str = "ec1868068cf40b0970f864c170130e8d"
    test_size: float = 0.3
    random_state: int = 42
    cache_dir: Optional[str] = None
    cache_expire_days: int = 7
    max_rounds_standings: int = 5
    max_rounds_position_analysis: int = 3
    
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), 'f1_cache')
        
        self.cache_path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

class CacheManager:
    def __init__(self, config: RaceConfig):
        self.config = config
        self.cache_path = config.cache_path
        self._setup_fastf1_cache()
    
    def _setup_fastf1_cache(self):
        """caching for session data"""
        fastf1_cache_dir = self.cache_path / 'fastf1_sessions'
        fastf1_cache_dir.mkdir(exist_ok=True)
        
        try:
            fastf1.Cache.enable_cache(str(fastf1_cache_dir))
            logger.info(f"FastF1 cache enabled at: {fastf1_cache_dir}")
        except Exception as e:
            logger.warning(f"Could not enable FastF1 cache: {e}")
    
    def clear_expired_cache(self):
        """remove cached data older than expire_days"""
        import time
        expire_seconds = self.config.cache_expire_days * 24 * 3600
        current_time = time.time()
        
        for cache_file in self.cache_path.rglob('*'):
            if cache_file.is_file():
                try:
                    if current_time - cache_file.stat().st_mtime > expire_seconds:
                        cache_file.unlink()
                        logger.debug(f"Removed expired cache file: {cache_file}")
                except Exception as e:
                    logger.debug(f"Could not remove cache file {cache_file}: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """get cache directory statistics"""
        try:
            files = list(self.cache_path.rglob('*'))
            total_files = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return {
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_path)
            }
        except Exception:
            return {'total_files': 0, 'total_size_mb': 0, 'cache_dir': str(self.cache_path)}

class F1DataExtractor:
    def __init__(self, config: RaceConfig):
        self.config = config
        self.cache_manager = CacheManager(config)
        self._standings_cache = None
        self._driver_teams_cache = None
        
        # Clean up old cache files
        self.cache_manager.clear_expired_cache()
        
        # Log cache status
        cache_stats = self.cache_manager.get_cache_stats()
        logger.info(f"Cache initialized: {cache_stats['total_files']} files, "
                   f"{cache_stats['total_size_mb']} MB")
        
    def extract_current_standings(self) -> Dict[str, float]:
        """Extract current season championship standings"""
        if self._standings_cache is not None:
            return self._standings_cache
            
        try:
            logger.info("Extracting current season standings...")
            standings = {}
            rounds_processed = 0
            
            for round_num in range(1, self.config.max_rounds_standings + 1):
                try:
                    session = fastf1.get_session(self.config.year, round_num, 'R')
                    session.load()
                    results = session.results
                    rounds_processed += 1
                    
                    for _, row in results.iterrows():
                        driver = row['Abbreviation']
                        team = row['TeamName']
                        points = row.get('Points', 0) if pd.notna(row.get('Points', 0)) else 0
                        
                        if driver not in standings:
                            standings[driver] = {'points': 0, 'team': team}
                        standings[driver]['points'] += points
                        standings[driver]['team'] = team
                        
                except Exception as e:
                    logger.debug(f"Failed to load round {round_num}: {e}")
                    continue
                    
            if standings and rounds_processed > 0:
                max_points = max(data['points'] for data in standings.values())
                team_points = {}
                for driver_data in standings.values():
                    team = driver_data['team']
                    if team not in team_points:
                        team_points[team] = 0
                    team_points[team] += driver_data['points']
                
                max_team_points = max(team_points.values()) if team_points else 1
                self._standings_cache = {team: points / max_team_points for team, points in team_points.items()}
                logger.info(f"Processed {rounds_processed} rounds for standings")
                return self._standings_cache
                
        except Exception as e:
            logger.warning(f"Could not extract current standings: {e}")
            
        return self._fallback_team_performance()
    
    def extract_driver_teams(self) -> Dict[str, str]:
        """Extract current driver-team assignments"""
        if self._driver_teams_cache is not None:
            return self._driver_teams_cache
            
        try:
            logger.info("Extracting driver-team mappings...")
            session = fastf1.get_session(self.config.year, 1, 'R')
            session.load()
            results = session.results
            
            driver_teams = {}
            for _, row in results.iterrows():
                driver_teams[row['Abbreviation']] = row['TeamName']
                
            self._driver_teams_cache = driver_teams
            logger.info(f"Extracted teams for {len(driver_teams)} drivers")
            return driver_teams
            
        except Exception as e:
            logger.warning(f"Could not extract driver teams: {e}")
            return self._fallback_driver_teams()
    
    def calculate_position_change_stats(self) -> Dict[str, float]:
        """Calculate average position change from recent races"""
        try:
            logger.info("Calculating position change statistics...")
            position_changes = {}
            race_count = {}
            races_analyzed = 0
            
            start_round = max(1, self.config.max_rounds_position_analysis - 2)
            end_round = self.config.max_rounds_position_analysis + 1
            
            for round_num in range(start_round, end_round):
                try:
                    session = fastf1.get_session(self.config.year, round_num, 'R')
                    session.load()
                    
                    quali_session = fastf1.get_session(self.config.year, round_num, 'Q')
                    quali_session.load()
                    
                    race_results = session.results.set_index('Abbreviation')['Position']
                    quali_results = quali_session.results.set_index('Abbreviation')['Position']
                    races_analyzed += 1
                    
                    for driver in race_results.index:
                        if driver in quali_results.index:
                            change = quali_results[driver] - race_results[driver]
                            
                            if driver not in position_changes:
                                position_changes[driver] = 0
                                race_count[driver] = 0
                            
                            position_changes[driver] += change
                            race_count[driver] += 1
                            
                except Exception as e:
                    logger.debug(f"Failed to load round {round_num}: {e}")
                    continue
            
            result = {driver: changes / race_count[driver] 
                     for driver, changes in position_changes.items() 
                     if race_count[driver] > 0}
            
            logger.info(f"Analyzed {races_analyzed} races for position changes")
            return result
                   
        except Exception as e:
            logger.warning(f"Could not calculate position changes: {e}")
            return self._fallback_position_changes()

    def extract_clean_air_pace(self) -> Dict[str, float]:
        """Extract clean air race pace from practice sessions"""
        try:
            logger.info("Loading practice session data...")
            session = fastf1.get_session(self.config.year, self.config.gp_name, 'FP1')
            session.load()
            
            laps = session.laps[session.laps['LapTime'].notna()]
            clean_laps = laps[
                laps['PitInTime'].isna() & laps['PitOutTime'].isna()
            ].copy()
            
            grouped = (
                clean_laps
                .groupby(['Driver', 'Stint'])['LapTime']
                .agg(['count', 'mean'])
                .reset_index()
            )
            
            best_stints = {}
            for driver, sub in grouped.groupby('Driver'):
                long_runs = sub[sub['count'] >= self.config.min_stint_laps]
                chosen = long_runs.loc[long_runs['mean'].idxmin()] if not long_runs.empty else sub.loc[sub['count'].idxmax()]
                best_stints[driver] = chosen['mean'].total_seconds()
                
            logger.info(f"Extracted clean air pace for {len(best_stints)} drivers")
            return best_stints
        except Exception as e:
            logger.warning(f"Could not extract clean air pace: {e}")
            return self._fallback_clean_air_pace()
    
    def extract_qualifying_data(self) -> pd.DataFrame:
        """Extract qualifying times and positions"""
        try:
            logger.info("Loading qualifying session data...")
            session = fastf1.get_session(self.config.year, self.config.gp_name, 'Q')
            session.load()
            
            laps = session.laps[['Driver','LapTime']].dropna()
            best = laps.groupby('Driver')['LapTime'].min().reset_index()
            best['QualifyingTime (s)'] = best['LapTime'].dt.total_seconds()
            
            result = best.sort_values('QualifyingTime (s)').reset_index(drop=True)
            logger.info(f"Extracted qualifying data for {len(result)} drivers")
            return result
        except Exception as e:
            logger.warning(f"Could not extract qualifying data: {e}")
            return self._fallback_qualifying_data()
    
    def extract_historical_sector_data(self) -> pd.DataFrame:
        """Extract sector times from previous year for baseline performance"""
        try:
            logger.info("Loading historical sector data...")
            session = fastf1.get_session(self.config.year - 1, self.config.gp_name, "R")
            session.load()
            
            laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
            laps.dropna(inplace=True)
            
            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                laps[f"{col} (s)"] = laps[col].dt.total_seconds()
            
            sector_times = laps.groupby("Driver").agg({
                "Sector1Time (s)": "mean",
                "Sector2Time (s)": "mean", 
                "Sector3Time (s)": "mean",
                "LapTime (s)": "mean"
            }).reset_index()
            
            sector_times["TotalSectorTime (s)"] = (
                sector_times["Sector1Time (s)"] + 
                sector_times["Sector2Time (s)"] + 
                sector_times["Sector3Time (s)"]
            )
            
            logger.info(f"Extracted historical data for {len(sector_times)} drivers")
            return sector_times
        except Exception as e:
            logger.error(f"Could not extract historical data: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self) -> Tuple[float, float]:
        """Fetch weather forecast for race day"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                "lat": 45.5031824, "lon": -73.5698065,
                "appid": self.config.api_key, "units": "metric"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            forecast_time = f"{self.config.year}-06-15 11:00:00"
            forecast = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)
            
            if forecast:
                return forecast["pop"], forecast["main"]["temp"]
            return 0.0, 20.0
        except Exception as e:
            logger.warning(f"Weather API failed: {e}")
            return 0.0, 20.0
    
    def _fallback_team_performance(self) -> Dict[str, float]:
        team_points = {
            'Red Bull Racing': 0.4, 'Mercedes': 0.44, 'Scuderia Ferrari': 0.46, 'McLaren': 1.0,
            'Aston Martin': 0.04, 'Alpine': 0.03, 'RB': 0.08, 'Williams': 0.15, 'Haas': 0.07, 'Kick Sauber': 0.04
        }
        return team_points
    
    def _fallback_driver_teams(self) -> Dict[str, str]:
        return {
            "VER": "Red Bull Racing", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Scuderia Ferrari",
            "RUS": "Mercedes", "HAM": "Scuderia Ferrari", "GAS": "Alpine", "ALO": "Aston Martin",
            "TSU": "RB", "SAI": "Williams", "HUL": "Kick Sauber", "OCO": "Haas",
            "STR": "Aston Martin", "COL": "Alpine", "HAD": "RB", "LAW": "RB",
            "ANT": "Mercedes", "BOR": "Kick Sauber", "BEA": "Haas", "ALB": "Williams"
        }
    
    def _fallback_position_changes(self) -> Dict[str, float]:
        return {
            'VER': -1.0, 'NOR': -1.0, 'RUS': 2.0, 'HAM': -3.0, 'PIA': 1.0,
            'ALO': 0.0, 'STR': -2.0, 'GAS': -6.0, 'OCO': -8.0, 'HUL': -6.0,
            'TSU': 6.0, 'SAI': 4.0, 'ALB': 7.0, 'LEC': 8.0
        }
    
    def _fallback_clean_air_pace(self) -> Dict[str, float]:
        return {
            'ALB': 78.014042, 'ALO': 91.652083, 'VER': 83.112994,
            'RUS': 77.970636, 'NOR': 79.685467, 'HAM': 86.677615
        }
    
    def _fallback_qualifying_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Driver": ['RUS', 'VER', 'PIA', 'HAM', 'ALO', 'NOR'],
            "QualifyingTime (s)": [70.899, 71.059, 71.120, 71.526, 71.586, 71.599]
        })

class FeatureEngineer:
    def __init__(self, extractor: F1DataExtractor):
        self.extractor = extractor
        self._team_performance = None
        self._driver_teams = None
        self._position_changes = None
    
    def _get_team_performance(self) -> Dict[str, float]:
        if self._team_performance is None:
            self._team_performance = self.extractor.extract_current_standings()
        return self._team_performance
    
    def _get_driver_teams(self) -> Dict[str, str]:
        if self._driver_teams is None:
            self._driver_teams = self.extractor.extract_driver_teams()
        return self._driver_teams
    
    def _get_position_changes(self) -> Dict[str, float]:
        if self._position_changes is None:
            self._position_changes = self.extractor.calculate_position_change_stats()
        return self._position_changes
    
    def create_features(self, qualifying_data: pd.DataFrame, clean_air_pace: Dict[str, float], 
                       sector_data: pd.DataFrame, weather: Tuple[float, float]) -> pd.DataFrame:
        """Engineer features"""
        df = qualifying_data.copy()
        
        team_performance = self._get_team_performance()
        driver_teams = self._get_driver_teams()
        position_changes = self._get_position_changes()
        
        df["Team"] = df["Driver"].map(driver_teams)
        df["TeamPerformanceScore"] = df["Team"].map(team_performance)
        df["CleanAirRacePace (s)"] = df["Driver"].map(clean_air_pace)
        df["AveragePositionChange"] = df["Driver"].map(position_changes)
        
        df["RainProbability"], df["Temperature"] = weather
        
        df["QualifyingPosition"] = df.index + 1
        df["QualifyingAdvantage"] = 1 / (df["QualifyingPosition"] + 1)
        
        if not sector_data.empty:
            df = df.merge(sector_data[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
        
        if "CleanAirRacePace (s)" in df.columns:
            df["PaceDifferential"] = df["CleanAirRacePace (s)"] - df["CleanAirRacePace (s)"].mean()
        
        return df

class F1RacePredictor:
    def __init__(self, config: RaceConfig):
        self.config = config
        self.models = {
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=config.random_state)
        }
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.best_model = None
        self.feature_names = None
    
    def prepare_data(self, df: pd.DataFrame, target_data: pd.Series = None) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """Prepare features"""
        feature_cols = [
            "QualifyingTime (s)", "RainProbability", "Temperature", 
            "TeamPerformanceScore", "CleanAirRacePace (s)", "QualifyingAdvantage"
        ]
        
        optional_features = ["TotalSectorTime (s)", "PaceDifferential", "AveragePositionChange"]
        for feat in optional_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        self.feature_names = feature_cols
        X = df[feature_cols]
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        if target_data is not None:
            # Create a temporary DataFrame to align indices properly
            temp_df = df.copy()
            temp_df['target'] = temp_df['Driver'].map(target_data)
            
            # Find rows with valid target values
            valid_rows = temp_df['target'].notna()
            
            if valid_rows.sum() > 0:
                # Filter using boolean indexing on DataFrame rows
                valid_indices = valid_rows.values  # Convert to numpy boolean array
                X_scaled_train = X_scaled[valid_indices]
                y_train = temp_df.loc[valid_rows, 'target'].values
                filtered_df = temp_df.loc[valid_rows].drop('target', axis=1).reset_index(drop=True)
                
                logger.info(f"Training data: {len(y_train)} samples after removing NaN values")
                return X_scaled_train, y_train, filtered_df
            else:
                logger.warning("No valid target data available")
        
        # Return full dataset if no valid target data
        return X_scaled, None, df
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train multiple models and select best performer"""
        if len(y) < 3:
            logger.warning(f"Insufficient training data: {len(y)} samples")
            return {'ridge': {'test_mae': 0.0, 'cv_mae': 0.0}}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        results = {}
        best_score = float('inf')
        
        for name, model in self.models.items():
            try:
                # Cross-validation
                if len(y_train) >= 3:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(3, len(y_train)), 
                                              scoring='neg_mean_absolute_error')
                    cv_mae = -cv_scores.mean()
                else:
                    cv_mae = 0.0
                
                # Test performance
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_mae = mean_absolute_error(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                results[name] = {
                    'cv_mae': cv_mae,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse
                }
                
                if test_mae < best_score:
                    best_score = test_mae
                    self.best_model = model
                    
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                continue
        
        if self.best_model is None:
            self.best_model = self.models['ridge']
            logger.info("Using Ridge regression as fallback")
        
        logger.info(f"Best model: {type(self.best_model).__name__} with MAE: {best_score:.3f}")
        return results
    
    def predict_race_results(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for all drivers"""
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare prediction data (use all drivers)
        feature_cols = [col for col in self.feature_names if col in feature_data.columns]
        X_pred = feature_data[feature_cols]
        
        X_pred_imputed = self.imputer.transform(X_pred)
        X_pred_scaled = self.scaler.transform(X_pred_imputed)
        
        predictions = self.best_model.predict(X_pred_scaled)
        
        results = pd.DataFrame({
            'Driver': feature_data['Driver'],
            'PredictedRaceTime (s)': predictions
        })
        
        return results.sort_values('PredictedRaceTime (s)').reset_index(drop=True)
    
    def get_feature_importance(self) -> pd.Series:
        """Extract feature importance from trained model"""
        if hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(self.best_model.feature_importances_, index=self.feature_names)
        elif hasattr(self.best_model, 'coef_'):
            return pd.Series(np.abs(self.best_model.coef_), index=self.feature_names)
        return pd.Series()

class RaceAnalyzer:
    @staticmethod
    def visualize_predictions(results: pd.DataFrame, clean_air_pace: Dict[str, float]):
        """Create visualization of race predictions"""
        results['CleanAirPace'] = results['Driver'].map(clean_air_pace)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Race time vs clean air pace
        ax1.scatter(results['CleanAirPace'], results['PredictedRaceTime (s)'])
        for _, row in results.iterrows():
            ax1.annotate(row['Driver'], 
                        (row['CleanAirPace'], row['PredictedRaceTime (s)']),
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Clean Air Race Pace (s)')
        ax1.set_ylabel('Predicted Race Time (s)')
        ax1.set_title('Clean Air Pace vs Predicted Race Time')
        
        # Predicted finishing positions
        ax2.barh(range(len(results)), results['PredictedRaceTime (s)'])
        ax2.set_yticks(range(len(results)))
        ax2.set_yticklabels(results['Driver'])
        ax2.set_xlabel('Predicted Race Time (s)')
        ax2.set_title('Predicted Race Results')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_results(results: pd.DataFrame, model_results: Dict[str, float], 
                       feature_importance: pd.Series):
        """Display race prediction results"""
        print("Race Prediction Results\n")
        
        # Podium prediction
        podium = results.head(3)
        print("PREDICTED PODIUM ")
        print(f"P1: {podium.iloc[0]['Driver']} ({podium.iloc[0]['PredictedRaceTime (s)']:.3f}s)")
        print(f"P2: {podium.iloc[1]['Driver']} ({podium.iloc[1]['PredictedRaceTime (s)']:.3f}s)")
        print(f"P3: {podium.iloc[2]['Driver']} ({podium.iloc[2]['PredictedRaceTime (s)']:.3f}s)")
        
        print(f"\n Full Results:")
        for i, row in results.iterrows():
            print(f"P{i+1:2d}: {row['Driver']} - {row['PredictedRaceTime (s)']:.3f}s")
        
        print(f"\n Model Performance:")
        for model_name, metrics in model_results.items():
            print(f"{model_name.upper()}: Test MAE = {metrics['test_mae']:.3f}s, "
                  f"CV MAE = {metrics['cv_mae']:.3f}s")
        
        if not feature_importance.empty:
            print(f"\n Top Feature Importances:")
            top_features = feature_importance.sort_values(ascending=False).head(5)
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.3f}")

def main():
    config = RaceConfig()
    
    extractor = F1DataExtractor(config)
    engineer = FeatureEngineer(extractor)
    predictor = F1RacePredictor(config)
    
    logger.info("Extracting race data...")
    qualifying_data = extractor.extract_qualifying_data()
    clean_air_pace = extractor.extract_clean_air_pace()
    sector_data = extractor.extract_historical_sector_data()
    weather = extractor.get_weather_data()
    
    logger.info("Engineering features...")
    feature_data = engineer.create_features(qualifying_data, clean_air_pace, sector_data, weather)
    
    target_data = None
    if not sector_data.empty:
        target_data = sector_data.set_index("Driver")["LapTime (s)"]
    
    # Prepare training data
    X_train, y_train, training_df = predictor.prepare_data(feature_data, target_data)
    
    if y_train is not None:
        logger.info("Training models...")
        model_results = predictor.train_and_evaluate(X_train, y_train)
    else:
        logger.info("No historical data available, using Ridge regression...")
        # Train on available data without target
        X_all, _, _ = predictor.prepare_data(feature_data, None)
        predictor.best_model = predictor.models['ridge']
        # Fit with dummy target for prediction pipeline
        dummy_target = np.ones(len(X_all))
        predictor.best_model.fit(X_all, dummy_target)
        model_results = {'ridge': {'test_mae': 0.0, 'cv_mae': 0.0}}
    
    logger.info("Generating predictions...")
    race_results = predictor.predict_race_results(feature_data)
    feature_importance = predictor.get_feature_importance()
    
    RaceAnalyzer.display_results(race_results, model_results, feature_importance)
    RaceAnalyzer.visualize_predictions(race_results, clean_air_pace)

if __name__ == "__main__":
    main()