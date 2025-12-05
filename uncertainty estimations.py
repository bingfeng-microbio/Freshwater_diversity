import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Define paths
model_save_path = 'RF_models'  # Directory to save trained Random Forest models
global_feature_file = 'data_for_prediction.csv'  # Global feature dataset path
tuning_results_file = 'hyperparameter_tuning.xlsx'  # Path to hyperparameter tuning results
output_folder = "uncertainty_results"  # Output directory for results

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Load global feature data
global_data = pd.read_csv(global_feature_file)
X_global = global_data.iloc[:, 2:]  # Feature columns (from the third column onward)
coordinates = global_data.iloc[:, :2]  # Extract Longitude and Latitude columns

# Load optimal hyperparameters
tuning_results = pd.read_excel(tuning_results_file)

# List of target variables
target_variables = [
    "Shannon_bac", "Shannon_arc", "Shannon_euk", "Shannon_vir",
    "BC_bac", "BC_arc", "BC_euk", "BC_vir"
]

# Parameters for uncertainty analysis
n_iterations = 1000  # Number of iterations using different random seeds
random_seeds = np.random.randint(0, 10000, size=n_iterations)

# DataFrame to store normalized uncertainty across all target variables
category_point_uncertainties = pd.DataFrame(coordinates)

# Loop over each target variable
for target_variable in target_variables:
    print(f"Processing target variable: {target_variable}")

    # Retrieve best hyperparameters
    best_params = tuning_results[tuning_results['Target Variable'] == target_variable]
    if best_params.empty:
        print(f"\tNo hyperparameters found for {target_variable}, skipping.")
        continue

    best_params = eval(best_params.iloc[0]['Best Parameters'])  # Convert string to dictionary

    # Remove any prefix from parameter names
    adjusted_params = {key.split('__')[-1]: value for key, value in best_params.items()}

    # Load training data
    data = pd.read_excel('data_for_ML.xlsx')
    X = data.iloc[:, 1:7]  # Select feature columns
    y = data[target_variable]  # Select target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Store predictions from each seed
    global_predictions = []

    for seed in random_seeds:
        np.random.seed(seed)
        model = RandomForestRegressor(random_state=seed, **adjusted_params)
        model.fit(X_train, y_train)

        # Save model to RF_models folder
        model_filename = os.path.join(model_save_path, f"{target_variable}_seed_{seed}_model.pkl")
        dump(model, model_filename)
        print(f"\tModel saved: {model_filename}")

        # Predict on global dataset
        predictions = model.predict(X_global)
        global_predictions.append(predictions)

    # Convert list to NumPy array
    global_predictions = np.array(global_predictions)

    # Compute interquartile range (75th - 25th percentile)
    quantile_75 = np.percentile(global_predictions, 75, axis=0)
    quantile_25 = np.percentile(global_predictions, 25, axis=0)
    uncertainty_range = quantile_75 - quantile_25

    # Save uncertainty results for the current variable
    results = pd.DataFrame({
        'Longitude': coordinates['Longitude'],
        'Latitude': coordinates['Latitude'],
        'Quantile 75%': quantile_75,
        'Quantile 25%': quantile_25,
        'Uncertainty Range': uncertainty_range
    })

    output_file = os.path.join(output_folder, f'{target_variable}_uncertainty.csv')
    results.to_csv(output_file, index=False)
    print(f"\tUncertainty analysis completed. Results saved to {output_file}")

    # Normalize uncertainty range to 0–1 scale
    normalized_uncertainty = (uncertainty_range - uncertainty_range.min()) / (uncertainty_range.max() - uncertainty_range.min())
    category_point_uncertainties[f'{target_variable}_Normalized_Uncertainty'] = normalized_uncertainty

# Save summary of normalized uncertainties at each location
point_uncertainty_file = os.path.join(output_folder, "uncertainty_summary_normalized.csv")
category_point_uncertainties.to_csv(point_uncertainty_file, index=False)
print(f"Normalized uncertainty analysis completed. Summary saved to {point_uncertainty_file}")
