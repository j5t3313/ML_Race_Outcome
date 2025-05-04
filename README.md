# Formula1RacePredictionModel

F1 Predictions 2025 - Machine Learning Model This project uses machine learning, FastF1 API data, and historical F1 race results to predict race outcomes for the 2025 Formula 1 season.

Project Overview:<br/>
This repository contains a Gradient Boosting Machine Learning model that predicts race results based on past performance, qualifying times, and other structured F1 data. 

The model leverages:<br/>

FastF1 API for historical race data 2024 race results 2025 qualifying session results<br/>

Data Sources <br/>
FastF1 API: Fetches lap times, race results, and telemetry data <br/>
2025 Qualifying Data: Used for prediction <br/>
Historical F1 Results: Processed from FastF1 for training the model<br/> 
Open Weather API: Weather forecast data

How It Works<br/>
Data Collection: The script pulls relevant F1 data using the FastF1 API. <br/>
Preprocessing & Feature Engineering: Converts lap times, normalizes driver names, and structures race data. <br/>
Model Training: A Gradient Boosting Regressor is trained using 2024 race results. <br/>
Prediction: The model predicts race times for 2025 and ranks drivers accordingly. <br/>
Evaluation: Model performance is measured using Mean Absolute Error (MAE).<br/> 
Dependencies: fastf1 numpy pandas scikit-learn matplotlib
