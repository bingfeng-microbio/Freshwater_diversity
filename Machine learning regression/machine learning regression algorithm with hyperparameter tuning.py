import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import numpy as np

# Load dataset
file_path = 'data_for_ML.xlsx'
data = pd.read_excel(file_path)

# Define feature matrix X and target variable columns
X = data.iloc[:, 1:7]
y_columns = data.columns[7:]

# Define models and corresponding hyperparameter grids
models = {
    'XGBoost': XGBRegressor(),
    'Random Forest': RandomForestRegressor(),
    'K Nearest Neighbors': KNeighborsRegressor()
}

param_grids = {
    'XGBoost': {
        'xgboost__n_estimators': [100, 200, 300, 500, 600],
        'xgboost__max_depth': [3, 5, 7, 9],
        'xgboost__learning_rate': [0.01, 0.1, 0.2, 0.3]
    },
    'Random Forest': {
        'randomforest__n_estimators': [100, 200, 300, 500, 600],
        'randomforest__max_depth': [None, 10, 20, 30, 40],
        'randomforest__min_samples_split': [2, 5]
    },
    'K Nearest Neighbors': {
        'knearestneighbors__n_neighbors': [3, 5, 7, 10],
        'knearestneighbors__weights': ['uniform', 'distance'],
        'knearestneighbors__metric': ['euclidean', 'manhattan']
    }
}

results = []
all_results = []
feature_importances = []
model_save_path = 'models2'
os.makedirs(model_save_path, exist_ok=True)

# Iterate over each target variable
for y_column in y_columns:
    print(y_column)
    y = data[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    for model_name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), (model_name.lower().replace(' ', ''), model)])
        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=10, scoring='r2', refit=True, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Save full grid search results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results['model_name'] = model_name
        cv_results['target_variable'] = y_column
        all_results.append(cv_results)

        # Extract best hyperparameters
        best_params = grid_search.best_params_

        # Evaluate on test set
        y_pred_test = grid_search.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # Evaluate on full dataset
        y_pred_overall = grid_search.predict(X)
        r2_overall = r2_score(y, y_pred_overall)
        rmse_overall = np.sqrt(mean_squared_error(y, y_pred_overall))

        # Save the best model
        model_filename = f"{model_save_path}/{y_column}_{model_name.replace(' ', '_')}_best_model.pkl"
        dump(grid_search.best_estimator_, model_filename)

        # Extract feature importances if the model is Random Forest
        if model_name == 'Random Forest':
            best_rf_model = grid_search.best_estimator_.named_steps['randomforest']
            importances = best_rf_model.feature_importances_
            for feature, importance in zip(X.columns, importances):
                feature_importances.append({
                    'Target Variable': y_column,
                    'Feature': feature,
                    'Importance': importance
                })

        # Append final evaluation metrics
        results.append({
            'Target Variable': y_column,
            'Model': model_name,
            'Best Parameters': str(best_params),
            'R2_overall': r2_overall,
            'RMSE_overall': rmse_overall,
            'R2_test': r2_test,
            'RMSE_test': rmse_test
        })

# Export results to Excel
results_df = pd.DataFrame(results)
feature_importances_df = pd.DataFrame(feature_importances)

with pd.ExcelWriter('hyperparameter_tuning2.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Model Results', index=False)
    feature_importances_df.to_excel(writer, sheet_name='Feature Importances', index=False)