from osgeo import gdal
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed

# 打开热浪堆叠的文件
dataset = gdal.Open(r"G:\粗分辨率补充\1堆叠\Air堆叠平滑.tif")

# 获取热浪堆叠文件的列数和行数
cols = dataset.RasterXSize
rows = dataset.RasterYSize

band = dataset.GetRasterBand(1)  # GDAL的波段索引从1开始
data = band.ReadAsArray()

projection = dataset.GetProjection()  # 投影
geotrans = dataset.GetGeoTransform()  # 几何信息

# 创建一个全零的数组，方便后续直接返回
zero_array = np.zeros(rows)

def process_column(col):
    # 读取当前列的数据
    column_data = data[:, col]  # 读取整列数据
    Original_data = column_data.copy()  # 备份原始数据

    # 跳过含有多个无效值的列
    if np.all(column_data[:3] < -100):
        return zero_array

    # 参数p值可以灵活设置
    p = 0.9  # 百分位值
    mean = np.mean(column_data)
    std_dev = np.std(column_data)

    if std_dev == 0:  # 防止标准差为0时出错
        return zero_array

    # 计算指定p值的正态分布分位点（阈值）
    Threshold = norm.ppf(p, mean, std_dev)

    # 高于阈值的区域标记为1，低于则为0
    Sort = np.where(Original_data > Threshold, 1, 0)

    # 初始化thing数组，标记3个连续的高温区域
    thing = np.zeros(len(Sort))
    windows = np.ones(3).astype(int)

    # 滑动窗口法标记连续3个高温的区域
    for i in range(len(Sort) - 2):
        sample = Sort[i:i + 3]
        if np.array_equal(sample, windows):
            thing[i:i + 3] = windows

    return thing

# 使用并行计算处理每一列数据
Final_data = Parallel(n_jobs=-1)(delayed(process_column)(col) for col in tqdm(range(cols)))

# 整合数据并输出(去趋势化)
result_all_new = np.transpose(np.array(Final_data), (1, 0))  # 转置恢复原来的行列顺序
print(result_all_new.shape)

# 将结果写入新的GeoTIFF文件
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(r"G:\粗分辨率补充\3高温干旱识别\高温识别平滑.tif", cols, rows, 1, gdal.GDT_Float32)
output_dataset.SetGeoTransform(geotrans)
output_dataset.SetProjection(projection)
output_dataset.GetRasterBand(1).WriteArray(result_all_new)
output_dataset = None  # 关闭文件
