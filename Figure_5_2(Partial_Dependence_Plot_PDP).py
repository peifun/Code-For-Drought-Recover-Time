import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from scipy.interpolate import make_interp_spline
import numpy as np
from sklearn.utils import resample

# 设置字体为 Times New Roman
plt.rc('font', family='Times New Roman')


# 异常值筛选函数（针对特定列）
def remove_outliers(df, columns):
    # 筛选指定列的异常值
    for column in columns:
        mean_val = df[column].mean()
        std_val = df[column].std()
        df = df[(df[column] >= mean_val - 3 * std_val) & (df[column] <= mean_val + 3 * std_val)]
    return df


# 曲线平滑函数，使用插值+移动平均
def smooth_curve(x, y, points=800, window_size=20):
    x_new = np.linspace(x.min(), x.max(), points)  # 在指定范围生成更多点
    spl = make_interp_spline(x, y, k=3)  # 使用三次样条插值
    y_smooth = spl(x_new)

    # 使用移动平均平滑曲线
    y_smooth = np.convolve(y_smooth, np.ones(window_size) / window_size, mode='same')

    return x_new, y_smooth


# 1. 数据加载
data_path = r"G:\PNAS子刊文件整理\7随机森林结果\非气候变量与残差.xlsx"
data = pd.read_excel(data_path)

# 2. 定义变量
x_columns = ['PF_ROOT', 'NF_ROOT', 'PF_TD', 'NF_TD',
             'PF_GPP', 'NF_GPP','PF_TA', 'NF_TA', 'PF_MS', 'NF_MS',
             'PF_AGB', 'NF_AGB', 'PF_TH', 'NF_TH',  'PF_LAI', 'NF_LAI', 'PF_WUE', 'NF_WUE']
y_columns = ['PF_Residuals', 'NF_Residuals']

# 3. 对Residuals部分筛选异常值
data_filtered_residuals = remove_outliers(data, y_columns)

# 4. 对所有变量筛选异常值
data_filtered_all = remove_outliers(data_filtered_residuals, x_columns + y_columns)

# 5. 模型训练
models = {}
for y_col in y_columns:
    X = data_filtered_all[x_columns]
    y = data_filtered_all[y_col]

    # 随机森林模型训练
    rf_model = RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=42)
    rf_model.fit(X, y)
    models[y_col] = rf_model


# 部分依赖曲线生成并添加95%置信区间
def compute_partial_dependence_with_ci(model, X, feature, n_bootstrap=100, ci=0.95):

    # 部分依赖的基础值
    pd_results = partial_dependence(model, X=X, features=[feature])
    x_vals = pd_results['values'][0]
    y_vals = pd_results['average'][0]

    # 引导法计算上下置信区间
    y_samples = []
    for _ in range(n_bootstrap):
        X_resampled = resample(X)  # 重采样数据
        pd_results_bootstrap = partial_dependence(model, X=X_resampled, features=[feature])
        y_samples.append(pd_results_bootstrap['average'][0])

    y_samples = np.array(y_samples)
    lower_percentile = (1 - ci) / 2 * 100  # 下边界百分位
    upper_percentile = (1 + ci) / 2 * 100  # 上边界百分位
    y_lower = np.percentile(y_samples, lower_percentile, axis=0)
    y_upper = np.percentile(y_samples, upper_percentile, axis=0)

    return x_vals, y_vals, y_lower, y_upper

# 6. 修改绘图，添加置信区间
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

# 计算所有子图的纵坐标范围
y_all = []
for i, (pf_feature, nf_feature) in enumerate(zip(x_columns[::2], x_columns[1::2])):
    # 获取 PF_Residuals 和 NF_Residuals 对应的部分依赖曲线值
    x_pf, y_pf, y_pf_lower, y_pf_upper = compute_partial_dependence_with_ci(
        models['PF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=pf_feature
    )
    x_nf, y_nf, y_nf_lower, y_nf_upper = compute_partial_dependence_with_ci(
        models['NF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=nf_feature
    )
    y_all.append((y_pf, y_nf))  # 添加每个子图的纵坐标数据

# 获取全局最小值和最大值
y_min = min([min(y_pf.min(), y_nf.min()) for y_pf, y_nf in y_all])
y_max = max([max(y_pf.max(), y_nf.max()) for y_pf, y_nf in y_all])

# 设置统一的纵坐标刻度
yticks = np.around(np.linspace(y_min, y_max, 5), decimals=2)

# 绘制所有子图
annotations = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
for i, (pf_feature, nf_feature) in enumerate(zip(x_columns[::2], x_columns[1::2])):
    ax = axes[i]

    # 绘制 PF_Residuals 对应的曲线
    x_pf, y_pf, y_pf_lower, y_pf_upper = compute_partial_dependence_with_ci(
        models['PF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=pf_feature
    )
    x_pf_smooth, y_pf_smooth = smooth_curve(x_pf, y_pf)  # 平滑曲线
    _, y_pf_lower_smooth = smooth_curve(x_pf, y_pf_lower)
    _, y_pf_upper_smooth = smooth_curve(x_pf, y_pf_upper)

    ax.plot(x_pf_smooth, y_pf_smooth, color=(109 / 255, 175 / 255, 215 / 255), label=f"{pf_feature}")
    ax.fill_between(x_pf_smooth, y_pf_lower_smooth, y_pf_upper_smooth, color=(109 / 255, 175 / 255, 215 / 255),
                    alpha=0.2)

    # 绘制 NF_Residuals 对应的曲线
    x_nf, y_nf, y_nf_lower, y_nf_upper = compute_partial_dependence_with_ci(
        models['NF_Residuals'],
        X=data_filtered_all[x_columns],
        feature=nf_feature
    )
    x_nf_smooth, y_nf_smooth = smooth_curve(x_nf, y_nf)  # 平滑曲线
    _, y_nf_lower_smooth = smooth_curve(x_nf, y_nf_lower)
    _, y_nf_upper_smooth = smooth_curve(x_nf, y_nf_upper)

    ax.plot(x_nf_smooth, y_nf_smooth, color=(253 / 255, 181 / 255, 118 / 255), label=f"{nf_feature}")
    ax.fill_between(x_nf_smooth, y_nf_lower_smooth, y_nf_upper_smooth, color=(253 / 255, 181 / 255, 118 / 255),
                    alpha=0.2)

    # 设置刻度
    ax.set_yticks(yticks)  # 使用统一的纵坐标刻度

    # 添加子图标注
    ax.text(0.02, 0.98, annotations[i], transform=ax.transAxes,
            fontsize=33, fontweight='bold', va='top', ha='left')

    # 设置图例
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

# 调整布局
plt.tight_layout()
plt.savefig(r"G:\PNAS子刊文件整理\7随机森林结果\补充实验\图\4_subplots_with_CI.png", dpi=300, bbox_inches="tight")
plt.show()
