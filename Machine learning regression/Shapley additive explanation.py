import os
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from joblib import load
import matplotlib.pyplot as plt

# 1. Load input data
file_path = 'data_for_ML.xlsx'
data = pd.read_excel(file_path)

# 2. Define features X and target variables y
X = data.iloc[:, 1:7]
y_columns = data.columns[7:]

# 3. Create directory to save SHAP outputs
model_save_path = 'models'
shap_save_path = 'shap_analysis'
os.makedirs(shap_save_path, exist_ok=True)

# 4. Compute SHAP values for each target variable
for y_column in y_columns:
    print(f"Calculating SHAP values for target: {y_column}")

    # 4.1 Get the current target variable
    y = data[y_column]

    # 4.2 Split data into training and test sets (X used entirely for SHAP explanation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 4.3 Load the trained model
    model_filename = f"{model_save_path}/{y_column}_Random_Forest_best_model.pkl"
    if os.path.exists(model_filename):
        best_model = load(model_filename)  # Load the full pipeline
        model = best_model.named_steps['randomforest']  # Extract the Random Forest model

        # 4.4 Apply any preprocessing transformation if the pipeline includes scalers
        if 'standardscaler' in best_model.named_steps:
            X_transformed = best_model.named_steps['standardscaler'].transform(X)
        elif 'minmaxscaler' in best_model.named_steps:
            X_transformed = best_model.named_steps['minmaxscaler'].transform(X)
        elif 'robustscaler' in best_model.named_steps:
            X_transformed = best_model.named_steps['robustscaler'].transform(X)
        else:
            X_transformed = X  # Use raw data if no scaler was used

        # 4.5 Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        # 4.6 Generate and save SHAP summary plot
        shap.summary_plot(shap_values, X_transformed, feature_names=X.columns, show=False)
        plt.title(f"SHAP Summary Plot for Random Forest ({y_column})")
        plt.savefig(f"{shap_save_path}/{y_column}_Random_Forest_shap_summary.png")
        plt.close()

        # 4.7 Save SHAP values to CSV
        shap_df = pd.DataFrame(shap_values, columns=X.columns)
        shap_df.to_csv(f"{shap_save_path}/{y_column}_Random_Forest_shap_values.csv", index=False)

        print(f"SHAP values computed and saved for Random Forest ({y_column})")

print("All SHAP analyses completed successfully!")