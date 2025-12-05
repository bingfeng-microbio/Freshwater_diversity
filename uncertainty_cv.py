import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# 定义路径
model_save_path = 'RF_models'  # 保存的最佳随机森林模型路径
global_feature_file = 'data_for_prediction.csv'  # 全球特征文件路径
tuning_results_file = 'hyperparameter_tuning.xlsx'  # 超参数结果文件
output_folder = "uncertainty_results_normalized"  # 输出路径

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 加载全球特征数据
global_data = pd.read_csv(global_feature_file)
coordinates = global_data.iloc[:, 1:3]  # 提取 Longitude 和 Latitude (第2和第3列)
X_global = global_data.iloc[:, 3:]      # 第4列开始是特征数据

# 加载最佳超参数
tuning_results = pd.read_excel(tuning_results_file)

# 目标变量列表
#target_variables = [
#    "Shannon_bac", "Shannon_arc", "Shannon_fungi", "Shannon_vir",
#    "BC_bac", "BC_arc", "BC_fungi", "BC_vir",
#    'Carbon_fixation','ANR','Denitrification','DNR','Nitrification','Nitrogen_fixation','Photosynthesis','ASR','DSR','Sulfur_oxidation'
#]
target_variables = [
    "Shannon_fungi", "BC_fungi"
]

# 不确定性分析参数
n_iterations = 1000  # 重新训练次数（使用 10 个随机种子）
random_seeds = np.random.randint(0, 10000, size=n_iterations)

# 保存结果
category_point_uncertainties = pd.DataFrame(coordinates)

# 对每个目标变量进行分析
for target_variable in target_variables:
    print(f"正在处理目标变量: {target_variable}")

    # 获取最佳超参数
    best_params = tuning_results[tuning_results['Target Variable'] == target_variable]
    if best_params.empty:
        print(f"\t未找到目标变量 {target_variable} 的最佳超参数，跳过。")
        continue

    best_params = eval(best_params.iloc[0]['Best Parameters'])  # 将字符串解析为字典

    # 去掉前缀
    adjusted_params = {key.split('__')[-1]: value for key, value in best_params.items()}

    # 加载数据进行模型训练
    data = pd.read_excel('data_for_ML.xlsx')
    X = data.iloc[:, 1:19]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 保存每次随机种子训练的预测值
    global_predictions = []

    for seed in random_seeds:
        np.random.seed(seed)
        model = RandomForestRegressor(random_state=seed, **adjusted_params)
        model.fit(X_train, y_train)

        # 保存模型到 RF_models 文件夹
        model_filename = os.path.join(model_save_path, f"{target_variable}_seed_{seed}_model.pkl")
        dump(model, model_filename)
        print(f"\t已保存模型: {model_filename}")

        # 使用模型预测
        predictions = model.predict(X_global)
        global_predictions.append(predictions)

    # 转换为 NumPy 数组
    global_predictions = np.array(global_predictions)

    # 计算变异系数 (Coefficient of Variation, CV)
    mean_predictions = np.mean(global_predictions, axis=0)
    std_predictions = np.std(global_predictions, axis=0)
    cv_uncertainty = std_predictions / (mean_predictions + 1e-10)  # 避免除以0

    # 保存每个目标变量的结果
    results = pd.DataFrame({
        'Hylak_id': global_data['Hylak_id'],  # 第一列ID
        'Longitude': coordinates['Longitude1'],
        'Latitude': coordinates['Latitude1'],
        'Mean_Prediction': mean_predictions,
        'Std_Prediction': std_predictions,
        'CV_Uncertainty': cv_uncertainty
    })

    # 保存每个目标变量的不确定性文件为 CSV
    output_file = os.path.join(output_folder, f'{target_variable}_uncertainty.csv')
    results.to_csv(output_file, index=False)
    print(f"\t不确定性分析完成，结果保存至 {output_file}")



