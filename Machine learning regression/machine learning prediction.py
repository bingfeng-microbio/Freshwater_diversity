import pandas as pd
from joblib import load
import os

# Load input data
input_file = 'data_for_prediction.csv'  # Input CSV file containing features
new_data = pd.read_csv(input_file)

# Read target identifiers (one per line) from a text file
with open('model_paths.txt', 'r') as file:
    target_identifiers = [line.strip() for line in file.readlines()]

# Extract feature columns (assumed to be from the 3rd to 8th column)
X_new = new_data.iloc[:, 2:8]

# Create output folder to store prediction results
output_folder = 'predictions_2023'
os.makedirs(output_folder, exist_ok=True)

# Perform prediction for each microbial target and save result
for identifier in target_identifiers:
    print(f"Processing model: {identifier}")
    model_path = os.path.join('models', f"{identifier}_Random_Forest_best_model.pkl")

    # Load the trained model
    model = load(model_path)

    # Generate predictions
    prediction_result = new_data.iloc[:, :2].copy()  # Preserve coordinate columns
    prediction_result[identifier] = model.predict(X_new)

    # Save prediction to CSV
    output_file = os.path.join(output_folder, f"{identifier}_prediction_RF.csv")
    prediction_result.to_csv(output_file, index=False)

    print(f"Prediction completed for model {identifier}. Saved to {output_file}")

print(f"All predictions completed! Results saved in the '{output_folder}' folder.")
