# Formula1RacePredictionModel
#
#
F1 Predictions 2025 - Machine Learning Model This project uses machine learning, FastF1 API data, and historical F1 race results to predict race outcomes for the 2025 Formula 1 season.
#
Project Overview:
This repository contains a Gradient Boosting Machine Learning model that predicts race results based on past performance, qualifying times, and other structured F1 data. The model leverages:

FastF1 API for historical race data 2024 race results 2025 qualifying session results Over the course of the season we will be adding additional data to improve our model as well Feature engineering techniques to improve predictions Data Sources FastF1 API: Fetches lap times, race results, and telemetry data 2025 Qualifying Data: Used for prediction Historical F1 Results: Processed from FastF1 for training the model How It Works Data Collection: The script pulls relevant F1 data using the FastF1 API. Preprocessing & Feature Engineering: Converts lap times, normalizes driver names, and structures race data. Model Training: A Gradient Boosting Regressor is trained using 2024 race results. Prediction: The model predicts race times for 2025 and ranks drivers accordingly. Evaluation: Model performance is measured using Mean Absolute Error (MAE). Dependencies: fastf1 numpy pandas scikit-learn matplotlib

Usage Run the prediction script:

python3 prediction1.py Expected output:

🏁 Predicted 2025 Miami GP Winner 🏁 Driver: Charles Leclerc, Predicted Race Time: 82.67s ... 🔍 Model Error (MAE): 3.22 seconds 📈 Model Performance The Mean Absolute Error (MAE) is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.

License This project is licensed under the MIT License.
