import rasterio
import numpy as np
from osgeo import gdal
from joblib import Parallel, delayed
from tqdm import tqdm

# 打开输入栅格文件
input_raster1 = rasterio.open(r"G:\粗分辨率补充\3高温干旱识别\纯干旱识别1.tif")
input_raster2 = rasterio.open(r"G:\粗分辨率补充\2去趋势\kNDVI去趋势.tif")
input_raster3 = gdal.Open(r"G:\粗分辨率补充\SPEI\SPEI_final\spei01_1213.tif")

# 获取栅格的列数和行数
cols = input_raster1.width
rows = input_raster1.height
cols2 = input_raster2.width
rows2 = input_raster2.height
print(cols, rows)
print(cols2, rows2)
raster_col = input_raster3.RasterXSize
raster_row = input_raster3.RasterYSize

# 读取栅格数据到内存中
data1 = input_raster1.read(1)
data2 = input_raster2.read(1)

# 创建输出数组
result = np.empty((1, cols))


# 定义计算异常值的函数
def calculate_anomaly(col_idx, data1, data2, num_years=20):
    col_data1 = data1[:, col_idx]
    col_data2 = data2[:, col_idx]

    # 检查该列是否包含干旱值
    if np.sum(col_data1) == 0:
        return 0

    # 获取干旱和非干旱月份的索引
    drought_months = np.where(col_data1 == 1)[0]
    non_drought_months = np.where(col_data1 == 0)[0]

    if len(drought_months) == 0 or len(non_drought_months) == 0:
        return 0

    # 计算干旱和非干旱月份的均值和标准差
    Drought_mean = np.mean(col_data2[drought_months])
    non_Drought_mean = np.mean(col_data2[non_drought_months])
    STD = np.std(col_data2[non_drought_months])

    if STD == 0:
        Anomaly = 0
    else:
        Anomaly = (Drought_mean - non_Drought_mean) / STD

    return Anomaly


# 并行计算每列的异常值
anomalies = Parallel(n_jobs=-1)(delayed(calculate_anomaly)(col, data1, data2) for col in tqdm(range(cols)))

# 转换和重塑异常值数组
result = np.array(anomalies).reshape((1, cols))
result = result.reshape(raster_row, raster_col)

# 创建输出栅格文件
output_tif_path = r"G:\粗分辨率补充\5干旱异常\1月异常.tif"
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(
    output_tif_path,
    raster_col,
    raster_row,
    1,
    gdal.GDT_Float32,
)
output_dataset.SetProjection(input_raster3.GetProjection())
output_dataset.SetGeoTransform(input_raster3.GetGeoTransform())
output_dataset.GetRasterBand(1).WriteArray(result)
output_dataset.FlushCache()
output_dataset = None
