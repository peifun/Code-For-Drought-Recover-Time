import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump, load

# 读取Excel数据
data = pd.read_excel(r"F:\计算代码\随机森林\全部变量.xlsx")

# 提取自变量和因变量
X = data[['RAT', 'RVPD', 'RSSR']]
y = data["NF_PF"]

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义要调优的超参数组合
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2']
}

param_grid2 = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2']
}

# 创建随机森林回归模型
model = RandomForestRegressor()

# 创建网格搜索对象
grid_search = GridSearchCV(model, param_grid2, cv=5, n_jobs=-1)

# 拟合模型
grid_search.fit(X, y)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 评估模型
score = grid_search.score(X, y)
print("模型得分:", score)

# # 保存模型到文件
# dump(grid_search, r'I:\子刊计算代码\随机森林/人工随机森林拟合.joblib')

# 输出特征的重要性
feature_importances = grid_search.best_estimator_.feature_importances_
for i, feature_name in enumerate(X.columns):
    print("特征:", feature_name, "重要性:", feature_importances[i])

