import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Read data
try:
    data = pd.read_excel(r"J:\\计算代码\\随机森林\\全部变量.xlsx")
except FileNotFoundError:
    print("File path error, please check if the file path is correct.")
    exit()
except KeyError:
    print("The specified sheet does not exist, please check the sheet_name parameter.")
    exit()

# Define subplot data, including color map parameter
subplots_data = [
    {"mat": data['NF_RAT'].values, "map": data['NF_RSSR'].values, "sensitivity_change": data['meanNF'].values,
     "title": "NF_RAT vs NF_RSSR", "xlabel": "NF_RSSR", "ylabel": "NF_RAT", "cmap": "Oranges"},
    {"mat": data['NF_RAT'].values, "map": data['NF_RVPD'].values, "sensitivity_change": data['meanNF'].values,
     "title": "NF_RAT vs NF_RVPD", "xlabel": "NF_RVPD", "ylabel": "NF_RAT", "cmap": "Oranges"},
    {"mat": data['PF_RAT'].values, "map": data['PF_RSSR'].values, "sensitivity_change": data['meanPF'].values,
     "title": "PF_RAT vs PF_RSSR", "xlabel": "PF_RSSR", "ylabel": "PF_RAT", "cmap": "Blues"},
    {"mat": data['PF_RAT'].values, "map": data['PF_RVPD'].values, "sensitivity_change": data['meanPF'].values,
     "title": "PF_RAT vs PF_RVPD", "xlabel": "PF_RVPD", "ylabel": "PF_RAT", "cmap": "Blues"}
]

# Create figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten subplot array for easier iteration

# Get x-axis range (NF_RSSR, PF_RSSR, NF_RVPD, PF_RVPD)
x_min1 = min(min(data['NF_RSSR'].values), min(data['PF_RSSR'].values))
x_max1 = max(max(data['NF_RSSR'].values), max(data['PF_RSSR'].values))

x_min2 = min(data['PF_RVPD'].values)
x_max2 = max(data['PF_RVPD'].values)

# Get sensitivity change range
vmin = 0
vmax = 18  # Set max value for color bar to 18

for idx, subplot in enumerate(subplots_data):
    mat = subplot["mat"]
    map = subplot["map"]
    sensitivity_change = subplot["sensitivity_change"]
    cmap = subplot["cmap"]

    # Check if data is empty
    if len(mat) == 0 or len(map) == 0 or len(sensitivity_change) == 0:
        print(f"Subplot {subplot['title']} data is empty, skipping plot.")
        continue

    # Create scatter plot (individual points)
    scatter = axes[idx].scatter(map, mat, c=sensitivity_change, cmap=cmap, vmin=vmin, vmax=vmax, s=30, edgecolors='none', alpha=0)

    # Set subplot title and axis labels
    axes[idx].set_xlabel(subplot["xlabel"], fontsize=14, family='Times New Roman')
    axes[idx].set_ylabel(subplot["ylabel"], fontsize=14, family='Times New Roman')
    axes[idx].tick_params(axis='both', labelsize=15)

    # Set uniform x-axis range for subplots
    if idx == 0 or idx == 2:
        axes[idx].set_xlim(x_min1, x_max1)  # First column subplots
    elif idx == 1 or idx == 3:
        axes[idx].set_xlim(x_min2, x_max2)  # Second column subplots
    num_bins_x = 24  # Number of bins for x-axis
    num_bins_y = 24  # Number of bins for y-axis

    # Get bin edges for the grid
    x_bins = np.linspace(axes[idx].get_xlim()[0], axes[idx].get_xlim()[1], num_bins_x + 1)
    y_bins = np.linspace(axes[idx].get_ylim()[0], axes[idx].get_ylim()[1], num_bins_y + 1)

    # Create 2D histogram to calculate mean within grid cells
    means = np.zeros((num_bins_y, num_bins_x))  # Array to store mean values

    for i in range(num_bins_x):
        for j in range(num_bins_y):
            # Define the range of each bin
            x_range = (x_bins[i], x_bins[i + 1])
            y_range = (y_bins[j], y_bins[j + 1])

            # Get indices of points within the current bin
            mask = (map >= x_range[0]) & (map < x_range[1]) & (mat >= y_range[0]) & (mat < y_range[1])

            # Calculate mean sensitivity_change for points in the current bin
            if np.any(mask):
                means[j, i] = np.mean(sensitivity_change[mask])
            else:
                means[j, i] = np.nan  # No data in the bin

    # Plot the grid-based mean values as an image (optional)
    cax = axes[idx].imshow(means, origin="lower", aspect="auto", cmap=cmap, extent=(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]))

    # Add color bar with range 0 to 18
    cbar = fig.colorbar(cax, ax=axes[idx], orientation="vertical", pad=0.03)
    cbar.set_label('RT', fontsize=14, family='Times New Roman')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(np.arange(vmin, vmax + 1, 2))  # Add tick intervals
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()

    # Add subplot labels
    labels = ['a', 'b', 'c', 'd']
    axes[idx].text(0.05, 0.9, labels[idx], transform=axes[idx].transAxes, fontsize=25, fontweight='bold', color='black', family='Times New Roman')

# Adjust layout
plt.tight_layout()
plt.savefig(r'G:\\PNAS子刊文件整理\\作图数据\\附件\\气候阶梯\\整体.png', dpi=500)  # 设置 dpi 参数，保存高分辨率图像
plt.show()
