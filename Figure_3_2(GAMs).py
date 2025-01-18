import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from pygam import GAM, s
import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib.gridspec as gridspec

# Set font to Times New Roman for all text
plt.rcParams.update({'font.family': 'Times New Roman'})

# 读取Excel文件中的数据
file_path = r"J:\计算代码\随机森林\全部变量外因内因版.xlsx"
df = pd.read_excel(file_path)

# 选择需要的列
df = df[['NF_PF', 'PF_RAT', 'NF_RAT','PF_RSSR', 'NF_RSSR','PF_RVPD', 'NF_RVPD','PF_RPRE', 'NF_RPRE','PF_DF', 'NF_DF','PF_DI', 'NF_DI','PF_SF', 'NF_SF']]  # 确保这里包含所需的列

# 按绝对值排序，得到接近0的10%范围
df_sorted_by_abs = df.reindex(df['NF_PF'].abs().sort_values().index)

# 确定取样数量
sample_size = int(len(df) * 0.1)
middle_10_percent = df_sorted_by_abs.iloc[:sample_size]

# 将数据分为正值和负值两部分
positive_values = df[df['NF_PF'] > 0].sort_values(by='NF_PF', ascending=False)
negative_values = df[df['NF_PF'] < 0].sort_values(by='NF_PF')

# 取正值部分的最大5%
top_5_percent_positive = positive_values.head(int(len(positive_values) * 0.1))
# 取负值部分的最小5%
bottom_5_percent_negative = negative_values.head(int(len(negative_values) * 0.1))

# 计算两个子图的坐标范围
x_min = min(df['PF_SF'].min(), top_5_percent_positive['PF_SF'].min(), bottom_5_percent_negative['PF_SF'].min())
x_max = max(df['PF_SF'].max(), top_5_percent_positive['PF_SF'].max(), bottom_5_percent_negative['PF_SF'].max())
y_min = min(df['NF_SF'].min(), top_5_percent_positive['NF_SF'].min(), bottom_5_percent_negative['NF_SF'].min())
y_max = max(df['NF_SF'].max(), top_5_percent_positive['NF_SF'].max(), bottom_5_percent_negative['NF_SF'].max())

# 计算 top_5_percent_positive 和 bottom_5_percent_negative 的 NF_PF 范围
top_5_percent_positive_min = top_5_percent_positive['NF_PF'].min()
bottom_5_percent_negative_max = bottom_5_percent_negative['NF_PF'].max()

# 创建自定义的colormap，定义新的范围
cmap = plt.cm.RdBu.reversed()
colors = cmap(np.linspace(0, 1, 256))

# 计算新范围的索引
new_range_end = int((top_5_percent_positive_min - bottom_5_percent_negative['NF_PF'].min()) / (top_5_percent_positive['NF_PF'].max() - bottom_5_percent_negative['NF_PF'].min()) * 256)
new_range_start = int((bottom_5_percent_negative_max - bottom_5_percent_negative['NF_PF'].min()) / (top_5_percent_positive['NF_PF'].max() - bottom_5_percent_negative['NF_PF'].min()) * 256)

# 替换该范围的颜色为白色
colors[new_range_start:new_range_end] = [1, 1, 1, 1]  # 设置为白色

# 创建新的colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# 绘图
plt.figure(figsize=(40, 18))

for i in range(1, 19):
    ax = plt.subplot(3, 6, i)
    # 加粗每个子图的边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)

def plot_scatter_with_regression(PF_SF, NF_SF, NF_PF, x_label, y_label, cmap, x_min, x_max, y_min, y_max, colorbar_label):
    # 线性回归
    X = PF_SF.values.reshape(-1, 1)
    y = NF_SF.values
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"{x_label}-{y_label} 的 R²:", r2)
    # 绘制散点图
    scatter = plt.scatter(PF_SF, NF_SF, c=NF_PF, cmap=cmap, s=50)
    plt.plot(PF_SF, y_pred, color='red', linewidth=2, label=f"R² = {r2:.2f}")
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    # 设置刻度线字体大小和刻度值与轴线的间距
    tick_fontsize = 25  # 刻度值字体大小
    tick_pad = 10       # 刻度值与方框之间的间距
    plt.tick_params(axis='x', labelsize=tick_fontsize, pad=tick_pad)  # x 轴刻度
    plt.tick_params(axis='y', labelsize=tick_fontsize, pad=tick_pad)  # y 轴刻度
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    # 设置 x 和 y 轴一致的刻度范围与间隔
    ticks = np.linspace(min(x_min, y_min), max(x_max, y_max), 4)  # 保证 x 和 y 一致
    plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
    plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
    # 添加颜色条
    colorbar = plt.colorbar(scatter, ax=plt.gca(), orientation='vertical', pad=0.02)
    # 设置颜色条的刻度数为 5
    ticks = np.linspace(NF_PF.min(), NF_PF.max(), 4)
    colorbar.set_ticks(ticks)  # 将刻度分为5个
    colorbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])  # 设置颜色条刻度标签为保留两位小数
    colorbar.set_label(colorbar_label, fontsize=18)
    # 调整颜色条刻度标签的大小
    colorbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度标签的字体大小
    # 添加 R² 文本
    plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=25, verticalalignment='top',
             horizontalalignment='left', color='black')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    return plt.gca()



# 读取数据
file_path = r"J:\计算代码\随机森林\全部变量外因内因版.xlsx"
sheet_name = "0.6"
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 提取预测变量和响应变量
variables1 = ['ΔRAT', 'ΔRVPD', 'ΔRSSR', 'ΔRPRE', 'ΔSF', 'ΔDF', 'ΔDI']
variables2 = ['ΔGPP', 'ΔTA', 'ΔAGB', 'ΔTH', 'ΔLAI', 'ΔTD', 'ΔROOT', 'ΔWUE', 'ΔMS']
response = data['NF_PF'].values

# 设置字体
font_path = 'C:\\Windows\\Fonts\\times.ttf'  # Times New Roman 字体路径
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()  # 全局设置字体为 Times New Roman

# 标准化处理
scaler = StandardScaler()

# 定义颜色和标签
line_colors1 = [(193 / 255, 40 / 255, 45 / 255), (215 / 255, 87 / 255, 40 / 255), (237 / 255, 134 / 255, 34 / 255),
                (246 / 255, 182 / 255, 107 / 255), (100 / 255, 90 / 255, 80 / 255), (142 / 255, 101 / 255, 95 / 255),
                (191 / 255, 160 / 255, 155 / 255)]  # 恢复期气候 & 干旱条件
line_colors2 = [(16 / 255, 82 / 255, 142 / 255), (52 / 255, 129 / 255, 134 / 255), (90 / 255, 179 / 255, 125 / 255),
                (131 / 255, 205 / 255, 103 / 255), (191 / 255, 160 / 255, 155 / 255), (100 / 255, 90 / 255, 80 / 255),
                (142 / 255, 101 / 255, 95 / 255), (50 / 255, 60 / 255, 100 / 255), (230 / 255, 170 / 255, 90 / 255)]
all_variables = [variables1, variables2]
all_line_colors = [line_colors1, line_colors2]

# 创建图像
fig = plt.figure(figsize=(25, 25))

gs = gridspec.GridSpec(6, 4, figure=fig)
# 在第一个子图 (3行6列, 第1个位置) 中绘制点图
ax1 = fig.add_subplot(gs[0:2, 0:4])
# 加载并处理数据
data_contrib = pd.read_excel(r"J:\计算代码\随机森林\全部变量外因内因版.xlsx", sheet_name="RF")
data_melted = data_contrib.melt(id_vars=['VAR'], var_name='Sample', value_name='Importance')
# 绘制点图
sns.stripplot(x='VAR', y='Importance', hue='Sample', data=data_melted, jitter=False, dodge=False,
              palette="RdBu", size=20, alpha=0.8, edgecolor='black', linewidth=0.5)
# 设置x轴标签和y轴标签
plt.ylabel('Importance(%)', fontsize=35, fontproperties=font_prop)
plt.xlabel(' ', fontsize=33, fontproperties=font_prop)
# 设置x轴和y轴的刻度大小
plt.tick_params(axis='x',rotation=45,  labelsize=30)
plt.tick_params(axis='y', labelsize=30)
# 设置图例
plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
           prop=fm.FontProperties(fname=font_path, size=30))
# 设置网格线
plt.grid(True, linestyle='--', alpha=0.6)
# 设置点图边框
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
# 添加子图标注
# plt.text(-0.12, 1.1, 'a', transform=plt.gca().transAxes,
#          fontsize=70, fontproperties=font_prop, weight='bold', va='top', ha='right')
# 添加横向的Mean折线图
mean_values = data_contrib['Mean']
mean_x = range(len(mean_values))
plt.plot(mean_x, mean_values, color=(193 / 255, 40 / 255, 45 / 255), label='Mean', linewidth=3, marker='o')


# 绘制第二个拟合子图
plt.subplot(6, 4, 9)
for idx, (variables, colors) in enumerate(zip([variables1], [line_colors1])):
    for i, var in enumerate(variables):
        # 获取数据并去除异常值
        X = data[var].values
        percentile_low = np.percentile(X, 2.5)
        percentile_high = np.percentile(X, 97.5)
        mask = (X >= percentile_low) & (X <= percentile_high)
        X_filtered = X[mask]
        Y_filtered = response[mask]
        # 标准化处理
        X_filtered = scaler.fit_transform(X_filtered.reshape(-1, 1)).flatten()
        # 拟合模型
        gam = GAM(s(0, n_splines=6, spline_order=3))
        gam.fit(X_filtered.reshape(-1, 1), Y_filtered)
        # 生成预测数据
        X_pred = np.linspace(X_filtered.min(), X_filtered.max(), 100).reshape(-1, 1)
        Y_pred = gam.predict(X_pred)
        confidence_intervals = gam.confidence_intervals(X_pred, width=0.95)
        plt.xlabel("X-axis Label", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=25)  # 主刻度标签字体大小
        plt.tick_params(axis='both', which='minor', labelsize=25)  # 次刻度标签字体大小
        # 绘图
        plt.axhline(0, color='gray', linestyle='--')
        plt.plot(X_pred, Y_pred, color=colors[i], label=f'{var}')
        plt.fill_between(X_pred.flatten(), confidence_intervals[:, 0], confidence_intervals[:, 1],
                         color=colors[i], alpha=0.2)

plt.xlabel('External factors', fontsize=25, fontproperties=font_prop)
plt.ylabel('ΔRT', fontsize=33, fontproperties=font_prop)
plt.legend(loc='upper right', fontsize=13, prop=fm.FontProperties(fname=font_path, size=14), frameon=False,
          borderaxespad=0.1, labelspacing=0.2, ncol=3, bbox_to_anchor=(0.67, 0.5, 0.35, 0.5))

# 绘制第三个拟合子图
plt.subplot(6, 4, 10)
for idx, (variables, colors) in enumerate(zip([variables2], [line_colors2])):
    for i, var in enumerate(variables):
        # 获取数据并去除异常值
        X = data[var].values
        percentile_low = np.percentile(X, 2.5)
        percentile_high = np.percentile(X, 97.5)
        mask = (X >= percentile_low) & (X <= percentile_high)
        X_filtered = X[mask]
        Y_filtered = response[mask]
        # 标准化处理
        X_filtered = scaler.fit_transform(X_filtered.reshape(-1, 1)).flatten()
        # 拟合模型
        gam = GAM(s(0, n_splines=6, spline_order=3))
        gam.fit(X_filtered.reshape(-1, 1), Y_filtered)
        # 生成预测数据
        X_pred = np.linspace(X_filtered.min(), X_filtered.max(), 100).reshape(-1, 1)
        Y_pred = gam.predict(X_pred)
        confidence_intervals = gam.confidence_intervals(X_pred, width=0.95)
        plt.tick_params(axis='both', which='major', labelsize=25)  # 主刻度标签字体大小
        plt.tick_params(axis='both', which='minor', labelsize=25)  # 次刻度标签字体大小
        # 绘图
        plt.axhline(0, color='gray', linestyle='--')
        plt.plot(X_pred, Y_pred, color=colors[i], label=f'{var}')
        plt.fill_between(X_pred.flatten(), confidence_intervals[:, 0], confidence_intervals[:, 1],
                         color=colors[i], alpha=0.2)

plt.xlabel('Internal factors', fontsize=25, fontproperties=font_prop)
# plt.ylabel('ΔRT', fontsize=33, fontproperties=font_prop)
plt.legend(loc='upper right', fontsize=13, prop=fm.FontProperties(fname=font_path, size=14), frameon=False,  # Removed frameon=True
          borderaxespad=0.1, labelspacing=0.2, ncol=3, bbox_to_anchor=(0.67, 0.5, 0.345, 0.5))


#标识大小位置
code_locx = -0.35
code__locy = 1.1
code_size = 70
# 创建RAT
plt.subplot(6, 4, 11)
ax1 = plot_scatter_with_regression(
    PF_SF=middle_10_percent['PF_RAT'],
    NF_SF=middle_10_percent['NF_RAT'],
    NF_PF=middle_10_percent['NF_PF'],
    x_label='PF_RAT',
    y_label='NF_RAT',
    cmap=plt.cm.RdBu.reversed(),
    x_min=min(df['PF_RAT'].min(), top_5_percent_positive['PF_RAT'].min(), bottom_5_percent_negative['PF_RAT'].min()),
    x_max=max(df['PF_RAT'].max(), top_5_percent_positive['PF_RAT'].max(), bottom_5_percent_negative['PF_RAT'].max()),
    y_min=min(df['NF_RAT'].min(), top_5_percent_positive['NF_RAT'].min(), bottom_5_percent_negative['NF_RAT'].min()),
    y_max=max(df['NF_RAT'].max(), top_5_percent_positive['NF_RAT'].max(), bottom_5_percent_negative['NF_RAT'].max()),
    colorbar_label='NF_PF'
)
# plt.text(code_locx, code__locy, 'd', transform=plt.gca().transAxes,
#          fontsize=code_size, fontproperties=font_prop, weight='bold', va='top', ha='right')
#创建RSSR
plt.subplot(6, 4, 13)
ax2 = plot_scatter_with_regression(
    PF_SF=middle_10_percent['PF_RSSR'],
    NF_SF=middle_10_percent['NF_RSSR'],
    NF_PF=middle_10_percent['NF_PF'],
    x_label='PF_RSSR',
    y_label='NF_RSSR',
    cmap=plt.cm.RdBu.reversed(),
    x_min=min(df['PF_RSSR'].min(), top_5_percent_positive['PF_RSSR'].min(), bottom_5_percent_negative['PF_RSSR'].min()),
    x_max=max(df['PF_RSSR'].max(), top_5_percent_positive['PF_RSSR'].max(), bottom_5_percent_negative['PF_RSSR'].max()),
    y_min=min(df['NF_RSSR'].min(), top_5_percent_positive['NF_RSSR'].min(), bottom_5_percent_negative['NF_RSSR'].min()),
    y_max=max(df['NF_RSSR'].max(), top_5_percent_positive['NF_RSSR'].max(), bottom_5_percent_negative['NF_RSSR'].max()),
    colorbar_label='NF_PF'
)

#创建RVPD
plt.subplot(6, 4, 15)
ax3 = plot_scatter_with_regression(
    PF_SF=middle_10_percent['PF_RVPD'],
    NF_SF=middle_10_percent['NF_RVPD'],
    NF_PF=middle_10_percent['NF_PF'],
    x_label='PF_RVPD',
    y_label='NF_RVPD',
    cmap=plt.cm.RdBu.reversed(),
    x_min=min(df['PF_RVPD'].min(), top_5_percent_positive['PF_RVPD'].min(), bottom_5_percent_negative['PF_RVPD'].min()),
    x_max=max(df['PF_RVPD'].max(), top_5_percent_positive['PF_RVPD'].max(), bottom_5_percent_negative['PF_RVPD'].max()),
    y_min=min(df['NF_RVPD'].min(), top_5_percent_positive['NF_RVPD'].min(), bottom_5_percent_negative['NF_RVPD'].min()),
    y_max=max(df['NF_RVPD'].max(), top_5_percent_positive['NF_RVPD'].max(), bottom_5_percent_negative['NF_RVPD'].max()),
    colorbar_label='NF_PF'
)

# 创建RPRE
plt.subplot(6, 4, 17)
ax4 = plot_scatter_with_regression(
    PF_SF=middle_10_percent['PF_RPRE'],
    NF_SF=middle_10_percent['NF_RPRE'],
    NF_PF=middle_10_percent['NF_PF'],
    x_label='PF_RPRE',
    y_label='NF_RPRE',
    cmap=plt.cm.RdBu.reversed(),
    x_min=min(df['PF_RPRE'].min(), top_5_percent_positive['PF_RPRE'].min(), bottom_5_percent_negative['PF_RPRE'].min()),
    x_max=max(df['PF_RPRE'].max(), top_5_percent_positive['PF_RPRE'].max(), bottom_5_percent_negative['PF_RPRE'].max()),
    y_min=min(df['NF_RPRE'].min(), top_5_percent_positive['NF_RPRE'].min(), bottom_5_percent_negative['NF_RPRE'].min()),
    y_max=max(df['NF_RPRE'].max(), top_5_percent_positive['NF_RPRE'].max(), bottom_5_percent_negative['NF_RPRE'].max()),
    colorbar_label='NF_PF'
)

# 创建DF
plt.subplot(6, 4, 19)
ax5 = plot_scatter_with_regression(
    PF_SF=middle_10_percent['PF_DF'],
    NF_SF=middle_10_percent['NF_DF'],
    NF_PF=middle_10_percent['NF_PF'],
    x_label='PF_DF',
    y_label='NF_DF',
    cmap=plt.cm.RdBu.reversed(),
    x_min=min(df['PF_DF'].min(), top_5_percent_positive['PF_DF'].min(), bottom_5_percent_negative['PF_DF'].min()),
    x_max=max(df['PF_DF'].max(), top_5_percent_positive['PF_DF'].max(), bottom_5_percent_negative['PF_DF'].max()),
    y_min=min(df['NF_DF'].min(), top_5_percent_positive['NF_DF'].min(), bottom_5_percent_negative['NF_DF'].min()),
    y_max=max(df['NF_DF'].max(), top_5_percent_positive['NF_DF'].max(), bottom_5_percent_negative['NF_DF'].max()),
    colorbar_label='NF_PF'
)

# 创建DI
plt.subplot(6, 4, 21)
ax6 = plot_scatter_with_regression(
    PF_SF=middle_10_percent['PF_DI'],
    NF_SF=middle_10_percent['NF_DI'],
    NF_PF=middle_10_percent['NF_PF'],
    x_label='PF_DI',
    y_label='NF_DI',
    cmap=plt.cm.RdBu.reversed(),
    x_min=min(df['PF_DI'].min(), top_5_percent_positive['PF_DI'].min(), bottom_5_percent_negative['PF_DI'].min()),
    x_max=max(df['PF_DI'].max(), top_5_percent_positive['PF_DI'].max(), bottom_5_percent_negative['PF_DI'].max()),
    y_min=min(df['NF_DI'].min(), top_5_percent_positive['NF_DI'].min(), bottom_5_percent_negative['NF_DI'].min()),
    y_max=max(df['NF_DI'].max(), top_5_percent_positive['NF_DI'].max(), bottom_5_percent_negative['NF_DI'].max()),
    colorbar_label='NF_PF'
)

# 创建SF
plt.subplot(6, 4, 23)
ax7 = plot_scatter_with_regression(
    PF_SF=middle_10_percent['PF_SF'],
    NF_SF=middle_10_percent['NF_SF'],
    NF_PF=middle_10_percent['NF_PF'],
    x_label='PF_SF',
    y_label='NF_SF',
    cmap=plt.cm.RdBu.reversed(),
    x_min=min(df['PF_SF'].min(), top_5_percent_positive['PF_SF'].min(), bottom_5_percent_negative['PF_SF'].min()),
    x_max=max(df['PF_SF'].max(), top_5_percent_positive['PF_SF'].max(), bottom_5_percent_negative['PF_SF'].max()),
    y_min=min(df['NF_SF'].min(), top_5_percent_positive['NF_SF'].min(), bottom_5_percent_negative['NF_SF'].min()),
    y_max=max(df['NF_SF'].max(), top_5_percent_positive['NF_SF'].max(), bottom_5_percent_negative['NF_SF'].max()),
    colorbar_label='NF_PF'
)

Xlager = 25
Ylager = 25
colorlager = 20
R2 = 25
# 大差异范围RSSR
plt.subplot(6, 4, 14)
big_difference_data = pd.concat([top_5_percent_positive, bottom_5_percent_negative])
# 线性回归
X = big_difference_data[['PF_RSSR']].values
y = big_difference_data['NF_RSSR'].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"big range的 R²:", r2)
scatter2 = plt.scatter(big_difference_data['PF_RSSR'], big_difference_data['NF_RSSR'], c=big_difference_data['NF_PF'], cmap=custom_cmap, s=50)
plt.plot(big_difference_data['PF_RSSR'], y_pred, color='red', linewidth=2)
x_min = min(df['PF_RSSR'].min(), top_5_percent_positive['PF_RSSR'].min(), bottom_5_percent_negative['PF_RSSR'].min())
x_max = max(df['PF_RSSR'].max(), top_5_percent_positive['PF_RSSR'].max(), bottom_5_percent_negative['PF_RSSR'].max())
y_min = min(df['NF_RSSR'].min(), top_5_percent_positive['NF_RSSR'].min(), bottom_5_percent_negative['NF_RSSR'].min())
y_max = max(df['NF_RSSR'].max(), top_5_percent_positive['NF_RSSR'].max(), bottom_5_percent_negative['NF_RSSR'].max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
ticks = np.linspace(axis_min, axis_max, 4)
plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel('PF_RSSR', fontsize=Xlager)
plt.ylabel('NF_RSSR', fontsize=Ylager)
plt.tick_params(axis='x', labelsize=Xlager,pad= 10)  # x 轴刻度标签大小
plt.tick_params(axis='y', labelsize=Ylager,pad= 10)  # y 轴刻度标签大小
plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=R2, verticalalignment='top', horizontalalignment='left', color='black')
colorbar2 = plt.colorbar(scatter2, ax=plt.gca(), orientation='vertical', pad=0.02)
colorbar2.set_label('NF_PF', fontsize=18)
ticks2 = np.linspace(big_difference_data['NF_PF'].min(), big_difference_data['NF_PF'].max(),4)
tick_labels2 = [f"{tick:.2f}" for tick in ticks2]
colorbar2.set_ticks(ticks2)
colorbar2.set_ticklabels(tick_labels2)
colorbar2.ax.tick_params(labelsize=colorlager)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=25)  # 设置刻度值字号

# 大差异范围RAT
plt.subplot(6, 4, 12)
big_difference_data = pd.concat([top_5_percent_positive, bottom_5_percent_negative])
# 线性回归
X = big_difference_data[['PF_RAT']].values
y = big_difference_data['NF_RAT'].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"big range的 R²:", r2)
scatter2 = plt.scatter(big_difference_data['PF_RAT'], big_difference_data['NF_RAT'], c=big_difference_data['NF_PF'], cmap=custom_cmap, s=50)
plt.plot(big_difference_data['PF_RAT'], y_pred, color='red', linewidth=2)
x_min = min(df['PF_RAT'].min(), top_5_percent_positive['PF_RAT'].min(), bottom_5_percent_negative['PF_RAT'].min())
x_max = max(df['PF_RAT'].max(), top_5_percent_positive['PF_RAT'].max(), bottom_5_percent_negative['PF_RAT'].max())
y_min = min(df['NF_RAT'].min(), top_5_percent_positive['NF_RAT'].min(), bottom_5_percent_negative['NF_RAT'].min())
y_max = max(df['NF_RAT'].max(), top_5_percent_positive['NF_RAT'].max(), bottom_5_percent_negative['NF_RAT'].max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
ticks = np.linspace(axis_min, axis_max, 4)
plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel('PF_RAT', fontsize=Xlager)
plt.ylabel('NF_RAT', fontsize=Ylager)
plt.tick_params(axis='x', labelsize=Xlager,pad= 10)  # x 轴刻度标签大小
plt.tick_params(axis='y', labelsize=Ylager,pad= 10)  # y 轴刻度标签大小
plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=R2, verticalalignment='top', horizontalalignment='left', color='black')
colorbar2 = plt.colorbar(scatter2, ax=plt.gca(), orientation='vertical', pad=0.02)
colorbar2.set_label('NF_PF', fontsize=18)
ticks2 = np.linspace(big_difference_data['NF_PF'].min(), big_difference_data['NF_PF'].max(),4)
tick_labels2 = [f"{tick:.2f}" for tick in ticks2]
colorbar2.set_ticks(ticks2)
colorbar2.set_ticklabels(tick_labels2)
colorbar2.ax.tick_params(labelsize=colorlager)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=25)  # 设置刻度值字号


# 大差异范围RVPD
plt.subplot(6, 4, 16)
big_difference_data = pd.concat([top_5_percent_positive, bottom_5_percent_negative])
# 线性回归
X = big_difference_data[['PF_RVPD']].values
y = big_difference_data['NF_RVPD'].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"big range的 R²:", r2)
scatter2 = plt.scatter(big_difference_data['PF_RVPD'], big_difference_data['NF_RVPD'], c=big_difference_data['NF_PF'], cmap=custom_cmap, s=50)
plt.plot(big_difference_data['PF_RVPD'], y_pred, color='red', linewidth=2)
x_min = min(df['PF_RVPD'].min(), top_5_percent_positive['PF_RVPD'].min(), bottom_5_percent_negative['PF_RVPD'].min())
x_max = max(df['PF_RVPD'].max(), top_5_percent_positive['PF_RVPD'].max(), bottom_5_percent_negative['PF_RVPD'].max())
y_min = min(df['NF_RVPD'].min(), top_5_percent_positive['NF_RVPD'].min(), bottom_5_percent_negative['NF_RVPD'].min())
y_max = max(df['NF_RVPD'].max(), top_5_percent_positive['NF_RVPD'].max(), bottom_5_percent_negative['NF_RVPD'].max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
ticks = np.linspace(axis_min, axis_max, 4)
plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel('PF_RVPD', fontsize=Xlager)
plt.ylabel('NF_RVPD', fontsize=Ylager)
plt.tick_params(axis='x', labelsize=Xlager,pad= 10)  # x 轴刻度标签大小
plt.tick_params(axis='y', labelsize=Ylager,pad= 10)  # y 轴刻度标签大小
plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=R2, verticalalignment='top', horizontalalignment='left', color='black')
colorbar2 = plt.colorbar(scatter2, ax=plt.gca(), orientation='vertical', pad=0.02)
colorbar2.set_label('NF_PF', fontsize=18)
ticks2 = np.linspace(big_difference_data['NF_PF'].min(), big_difference_data['NF_PF'].max(),4)
tick_labels2 = [f"{tick:.2f}" for tick in ticks2]
colorbar2.set_ticks(ticks2)
colorbar2.set_ticklabels(tick_labels2)
colorbar2.ax.tick_params(labelsize=colorlager)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=25)  # 设置刻度值字号

# 大差异范围RPRE
plt.subplot(6, 4, 18)
big_difference_data = pd.concat([top_5_percent_positive, bottom_5_percent_negative])
# 线性回归
X = big_difference_data[['PF_RPRE']].values
y = big_difference_data['NF_RPRE'].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"big range的 R²:", r2)
scatter2 = plt.scatter(big_difference_data['PF_RPRE'], big_difference_data['NF_RPRE'], c=big_difference_data['NF_PF'], cmap=custom_cmap, s=50)
plt.plot(big_difference_data['PF_RPRE'], y_pred, color='red', linewidth=2)
x_min = min(df['PF_RPRE'].min(), top_5_percent_positive['PF_RPRE'].min(), bottom_5_percent_negative['PF_RPRE'].min())
x_max = max(df['PF_RPRE'].max(), top_5_percent_positive['PF_RPRE'].max(), bottom_5_percent_negative['PF_RPRE'].max())
y_min = min(df['NF_RPRE'].min(), top_5_percent_positive['NF_RPRE'].min(), bottom_5_percent_negative['NF_RPRE'].min())
y_max = max(df['NF_RPRE'].max(), top_5_percent_positive['NF_RPRE'].max(), bottom_5_percent_negative['NF_RPRE'].max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
ticks = np.linspace(axis_min, axis_max, 4)
plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel('PF_RPRE', fontsize=Xlager)
plt.ylabel('NF_RPRE', fontsize=Ylager)
plt.tick_params(axis='x', labelsize=Xlager,pad= 10)  # x 轴刻度标签大小
plt.tick_params(axis='y', labelsize=Ylager,pad= 10)  # y 轴刻度标签大小
plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=R2, verticalalignment='top', horizontalalignment='left', color='black')
colorbar2 = plt.colorbar(scatter2, ax=plt.gca(), orientation='vertical', pad=0.02)
colorbar2.set_label('NF_PF', fontsize=18)
ticks2 = np.linspace(big_difference_data['NF_PF'].min(), big_difference_data['NF_PF'].max(),4)
tick_labels2 = [f"{tick:.2f}" for tick in ticks2]
colorbar2.set_ticks(ticks2)
colorbar2.set_ticklabels(tick_labels2)
colorbar2.ax.tick_params(labelsize=colorlager)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=25)  # 设置刻度值字号

# 大差异范围DF
plt.subplot(6, 4, 20)
big_difference_data = pd.concat([top_5_percent_positive, bottom_5_percent_negative])
# 线性回归
X = big_difference_data[['PF_DF']].values
y = big_difference_data['NF_DF'].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"big range的 R²:", r2)
scatter2 = plt.scatter(big_difference_data['PF_DF'], big_difference_data['NF_DF'], c=big_difference_data['NF_PF'], cmap=custom_cmap, s=50)
plt.plot(big_difference_data['PF_DF'], y_pred, color='red', linewidth=2)
x_min = min(df['PF_DF'].min(), top_5_percent_positive['PF_DF'].min(), bottom_5_percent_negative['PF_DF'].min())
x_max = max(df['PF_DF'].max(), top_5_percent_positive['PF_DF'].max(), bottom_5_percent_negative['PF_DF'].max())
y_min = min(df['NF_DF'].min(), top_5_percent_positive['NF_DF'].min(), bottom_5_percent_negative['NF_DF'].min())
y_max = max(df['NF_DF'].max(), top_5_percent_positive['NF_DF'].max(), bottom_5_percent_negative['NF_DF'].max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
ticks = np.linspace(axis_min, axis_max, 4)
plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel('PF_DF', fontsize=Xlager)
plt.ylabel('NF_DF', fontsize=Ylager)
plt.tick_params(axis='x', labelsize=Xlager,pad= 10)  # x 轴刻度标签大小
plt.tick_params(axis='y', labelsize=Ylager,pad= 10)  # y 轴刻度标签大小
plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=R2, verticalalignment='top', horizontalalignment='left', color='black')
colorbar2 = plt.colorbar(scatter2, ax=plt.gca(), orientation='vertical', pad=0.02)
colorbar2.set_label('NF_PF', fontsize=18)
ticks2 = np.linspace(big_difference_data['NF_PF'].min(), big_difference_data['NF_PF'].max(),4)
tick_labels2 = [f"{tick:.2f}" for tick in ticks2]
colorbar2.set_ticks(ticks2)
colorbar2.set_ticklabels(tick_labels2)
colorbar2.ax.tick_params(labelsize=colorlager)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=25)  # 设置刻度值字号

# 大差异范围DI
plt.subplot(6, 4, 22)
big_difference_data = pd.concat([top_5_percent_positive, bottom_5_percent_negative])
# 线性回归
X = big_difference_data[['PF_DI']].values
y = big_difference_data['NF_DI'].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"big range的 R²:", r2)
scatter2 = plt.scatter(big_difference_data['PF_DI'], big_difference_data['NF_DI'], c=big_difference_data['NF_PF'], cmap=custom_cmap, s=50)
plt.plot(big_difference_data['PF_DI'], y_pred, color='red', linewidth=2)
x_min = min(df['PF_DI'].min(), top_5_percent_positive['PF_DI'].min(), bottom_5_percent_negative['PF_DI'].min())
x_max = max(df['PF_DI'].max(), top_5_percent_positive['PF_DI'].max(), bottom_5_percent_negative['PF_DI'].max())
y_min = min(df['NF_DI'].min(), top_5_percent_positive['NF_DI'].min(), bottom_5_percent_negative['NF_DI'].min())
y_max = max(df['NF_DI'].max(), top_5_percent_positive['NF_DI'].max(), bottom_5_percent_negative['NF_DI'].max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
ticks = np.linspace(axis_min, axis_max, 4)
plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel('PF_DI', fontsize=Xlager)
plt.ylabel('NF_DI', fontsize=Ylager)
plt.tick_params(axis='x', labelsize=Xlager,pad= 10)  # x 轴刻度标签大小
plt.tick_params(axis='y', labelsize=Ylager,pad= 10)  # y 轴刻度标签大小
plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=R2, verticalalignment='top', horizontalalignment='left', color='black')
colorbar2 = plt.colorbar(scatter2, ax=plt.gca(), orientation='vertical', pad=0.02)
colorbar2.set_label('NF_PF', fontsize=18)
ticks2 = np.linspace(big_difference_data['NF_PF'].min(), big_difference_data['NF_PF'].max(),4)
tick_labels2 = [f"{tick:.2f}" for tick in ticks2]
colorbar2.set_ticks(ticks2)
colorbar2.set_ticklabels(tick_labels2)
colorbar2.ax.tick_params(labelsize=colorlager)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=25)  # 设置刻度值字号


# 大差异范围子图SF
plt.subplot(6, 4, 24)
big_difference_data = pd.concat([top_5_percent_positive, bottom_5_percent_negative])
# 线性回归
X = big_difference_data[['PF_SF']].values
y = big_difference_data['NF_SF'].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"big range的 R²:", r2)
scatter2 = plt.scatter(big_difference_data['PF_SF'], big_difference_data['NF_SF'], c=big_difference_data['NF_PF'], cmap=custom_cmap, s=50)
plt.plot(big_difference_data['PF_SF'], y_pred, color='red', linewidth=2)
x_min = min(df['PF_SF'].min(), top_5_percent_positive['PF_SF'].min(), bottom_5_percent_negative['PF_SF'].min())
x_max = max(df['PF_SF'].max(), top_5_percent_positive['PF_SF'].max(), bottom_5_percent_negative['PF_SF'].max())
y_min = min(df['NF_SF'].min(), top_5_percent_positive['NF_SF'].min(), bottom_5_percent_negative['NF_SF'].min())
y_max = max(df['NF_SF'].max(), top_5_percent_positive['NF_SF'].max(), bottom_5_percent_negative['NF_SF'].max())
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
ticks = np.linspace(axis_min, axis_max, 4)
plt.xticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 x 轴刻度
plt.yticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])  # 设置 y 轴刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel('PF_SF', fontsize=Xlager)
plt.ylabel('NF_SF', fontsize=Ylager)
plt.tick_params(axis='x', labelsize=Xlager,pad= 10)  # x 轴刻度标签大小
plt.tick_params(axis='y', labelsize=Ylager,pad= 10)  # y 轴刻度标签大小
plt.text(0.05, 0.95, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=R2, verticalalignment='top', horizontalalignment='left', color='black')
colorbar2 = plt.colorbar(scatter2, ax=plt.gca(), orientation='vertical', pad=0.02)
colorbar2.set_label('NF_PF', fontsize=18)
ticks2 = np.linspace(big_difference_data['NF_PF'].min(), big_difference_data['NF_PF'].max(),4)
tick_labels2 = [f"{tick:.2f}" for tick in ticks2]
colorbar2.set_ticks(ticks2)
colorbar2.set_ticklabels(tick_labels2)
colorbar2.ax.tick_params(labelsize=colorlager)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=25)  # 设置刻度值字号


# 显示图像
plt.tight_layout()
plt.savefig(r'G:\PNAS子刊文件整理\代码放置\06DI反证制图.png', dpi=400)  # 设置 dpi 参数，保存高分辨率图像
print(file_path)