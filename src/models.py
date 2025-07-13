"""
Module for training machine learning models for the data.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
import sys
import io


def train_model(df: pd.DataFrame, model_type: str = 'tabular', fine_tune: bool = False) -> None:
    """Train and fine-tune a model, then save it.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and the target variable.
        model_type (str): The type of model being trained ('tabular' or 'multimodal').
                          This affects the file names for saving logs and models.
        fine_tune (bool): Whether to fine-tune the model using GridSearchCV.

    Returns:
        None
    """

    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer

    log_file_path = f'models/{model_type}_grid_search_log.txt'
    model_file_path = f'models/{model_type}_model.pkl'

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if fine_tune:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    else:
        param_grid = {
            'n_estimators': [200],
            'learning_rate': [0.2],
            'max_depth': [5],
            'subsample': [1.0],
            'colsample_bytree': [1.0]
        }
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, enable_categorical=True)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, 
                                scoring='neg_mean_squared_error', verbose=3, n_jobs=1)
    
    grid_search.fit(X_train, y_train)
    
    sys.stdout = original_stdout
    output_content = output_buffer.getvalue()
    output_buffer.close()

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results_content = f'Root Mean Squared Error (RMSE): {rmse:.2f}\nR2 Score: {r2:.4f}'
    print(results_content)  

    with open(log_file_path, "w") as f:
        f.write(output_content)
        f.write("\nBest parameters:\n")
        f.write(str(grid_search.best_params_))
        f.write("\nBest score (neg_mean_squared_error):\n")
        f.write(str(grid_search.best_score_))
        f.write("\n\n" + results_content)  
    
    with open(model_file_path, 'wb') as f:
        pickle.dump(best_model, f)
