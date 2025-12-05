import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from itertools import combinations

# ================================
# Step 1: Load data
# ================================
train_file = "data_for_ML.xlsx"
predict_file = "data_for_prediction.csv"

train_df = pd.read_excel(train_file)
predict_df = pd.read_csv(predict_file)

# ================================
# Step 2: Select environmental variables
# ================================
env_vars = ["Lake_area","Res_time","Slope_100","Depth_avg","si10","sp","ssr","t2m","tp",
            "Cropland","Forest","Grassland","Urban_land","Bare_land","Elevation"]

train_env = train_df[env_vars].values
predict_env = predict_df[env_vars].values

# ================================
# Step 3: PCA
# ================================
pca_full = PCA()
train_pcs_full = pca_full.fit_transform(train_env)

# 选择累计方差达到 90% 的主成分数
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.searchsorted(cumulative_var, 0.99) + 1

print(f"PCA explained variance ratio: {pca_full.explained_variance_ratio_}")
print(f"Cumulative variance: {cumulative_var}")
print(f"Number of principal components chosen: {n_components}")

# 只保留前 n_components
pca = PCA(n_components=n_components)
train_pcs = pca.fit_transform(train_env)
predict_pcs = pca.transform(predict_env)

# ================================
# Step 4: Compute extrapolation ratio using 2D convex hulls
# ================================
pc_combos = list(combinations(range(n_components), 2))
outside_counts = np.zeros(predict_pcs.shape[0])

for idx1, idx2 in pc_combos:
    pc_pair_train = train_pcs[:, [idx1, idx2]]
    delaunay = Delaunay(pc_pair_train)  # 使用所有训练点生成 Delaunay 三角剖分
    outside = delaunay.find_simplex(predict_pcs[:, [idx1, idx2]]) < 0
    outside_counts += outside

# 外推比例 = 外部组合数 / 总组合数
predict_df["extrapolation_ratio"] = outside_counts / len(pc_combos)

# ================================
# Step 5: Save results
# ================================
output_file = "prediction_with_extrapolation.csv"
predict_df.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
