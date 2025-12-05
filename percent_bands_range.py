import pandas as pd

# 1. 读取数据
train_file = "data_for_ML.xlsx"
pred_file = "data_for_prediction.csv"

train_df = pd.read_excel(train_file)
pred_df = pd.read_csv(pred_file)

# 2. 定义训练特征（去掉纬度与经度正弦余弦）
features = [
    "Lake_area", "Res_time", "Slope_100", "Depth_avg",
    "si10", "sp", "ssr", "t2m", "tp",
    "Cropland", "Forest", "Grassland", "Urban_land", "Bare_land",
    "Elevation"
]

# 3. 计算训练集每个特征的最小值与最大值
ranges = train_df[features].agg(['min', 'max']).transpose()
ranges.columns = ['min', 'max']
print("=== 训练集各变量范围 ===")
print(ranges)

# 4. 用于记录每个预测点是否超出范围
extrapolation_matrix = pd.DataFrame(index=pred_df.index, columns=features)

for feature in features:
    f_min = ranges.loc[feature, 'min']
    f_max = ranges.loc[feature, 'max']
    extrapolation_matrix[feature] = (
        (pred_df[feature] < f_min) | (pred_df[feature] > f_max)
    )

# 5. 统计每个变量有多少点越界
variable_out_stats = extrapolation_matrix.sum().sort_values(ascending=False)
print("\n=== 每个变量越界的点数 ===")
print(variable_out_stats)

# 6. 统计每个预测点在哪些变量越界
pred_df['out_of_range_features'] = extrapolation_matrix.apply(
    lambda row: [f for f, v in row.items() if v], axis=1
)
pred_df['num_out_of_range'] = extrapolation_matrix.sum(axis=1)

# 6b. 统计每个预测点落在范围内的变量数量
pred_df['num_in_range'] = len(features) - pred_df['num_out_of_range']

print("\n=== 示例：前10个预测点的越界情况 ===")
print(pred_df[['Hylak_id', 'num_out_of_range', 'num_in_range', 'out_of_range_features']].head(10))

# 7. 提取所有越界点和在范围内的点
out_of_range_points = pred_df[pred_df['num_out_of_range'] > 0]
in_range_points = pred_df[pred_df['num_out_of_range'] == 0]

print(f"\n✅ 总预测点数: {len(pred_df)}")
print(f"⚠ 存在外推（至少1个变量越界）的点数: {len(out_of_range_points)}")
print(f"✅ 完全在范围内的点数: {len(in_range_points)}")

# 8. 保存结果为 CSV
out_of_range_points.to_csv("points_with_extrapolation.csv", index=False)
in_range_points.to_csv("points_without_extrapolation.csv", index=False)

print("\n✅ CSV 文件已保存：")
print("points_with_extrapolation.csv (越界点)")
print("points_without_extrapolation.csv (完全在范围内点)")
