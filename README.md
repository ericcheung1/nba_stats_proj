# NBA Salary Prediction Pipeline

## Project Overview

This project aims to predict the salary cap percentage of NBA players for the 2023-24 season based on historical data and player statistics. The pipeline includes data loading, feature engineering, model selection, training, and evaluation. Different machine learning models from regression-based to tree-based were explored and compared to find the most accurate predictors and models. Data collected from 'https://www.basketball-reference.com/' and 'https://hoopshype.com/'.

## Data

The primary dataset used for this project is `full_data.csv`, located in the `data/` directory. This file contains historical NBA player statistics and salary information, spanning from the 2020-21 through 2022-23 NBA seasons. One model was trained on an expanded dataset containing data spanning the 2018-19 through 2022-23 NBA seasons contained in the `fu_data_six_seasons.csv` file. The data was collected using the `data_pipeline.py` file in the `data/` directory and preprocessed using the `preprocessing.py ` file in the `model_fit_and_eval/` directory. Functions and code used to collect and clean data including the `scraper2.py` and `cleaner.py` files can be found in `src/` directory. And the `database.py` file in the `data/` directory can be used to recreate the SQLite database used to store the collected data.

## Feature Engineering

A feature was created before the fitting step in select models, the feature being:

* **Rookie Scale Contract Indicator:** A binary feature indicating if a player is in their first 3-4 seasons (on a rookie scale contract).

## Model Training and Evaluation

The `model_fit_and_eval/` directory contains experiments with various machine learning models. 

* **Linear Regression** A basic linear model with performance tested using `TimeSeriesSplit` cross-validation which serves as a baseline measure of prediction power.
* **Lasso Regression:** A linear model with L1 regularization used for prediction and feature selection. Hyperparameter tuning (alpha) was performed using cross-validation `TimeSeriesSplit`.
* **Random Forest Regression:** A non-linear ensemble model that was trained on the full feature set and  a reduced feature set (based on Lasso's feature importance). Hyperparameters (e.g., `n_estimators`, `max_depth`) were tuned using `GridSearchCV` with `TimeSeriesSplit`.

Key metrics for evaluation include:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

The trained models can also be saved directly from their respective .py files in the `model_fit_and_eval/` directory.

