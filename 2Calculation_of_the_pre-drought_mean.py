import numpy as np
from osgeo import gdal
from tqdm import tqdm
from joblib import Parallel, delayed

filename = gdal.Open(r"G:\粗分辨率补充\3高温干旱识别\纯干旱识别12.tif")

# 获取栅格的列数和行数
columns = filename.RasterXSize
rows = filename.RasterYSize
# 获取栅格的波段数
band_count = filename.RasterCount
# 获取栅格变换信息
transform = filename.GetGeoTransform()
# 读取第一个波段的数据
band = filename.GetRasterBand(1)
data = band.ReadAsArray()
array = data.astype(float)

#创建新数组接受结果
start_column = 0  # 起始列索引（从0开始）
end_column = columns  # 结束列索引（不包含）
# 523221 样本点
# 523222
# 523223
# 523224

GPP = gdal.Open(r"G:\粗分辨率补充\2去趋势\kNDVI去趋势.tif")

# 获取栅格的列数和行数
GPPcolumns = GPP.RasterXSize
GPProws = GPP.RasterYSize

# 获取栅格的波段数
GPPband_count = GPP.RasterCount

# 获取栅格变换信息
GPPtransform = GPP.GetGeoTransform()

# 读取第一个波段的数据
GPPband = GPP.GetRasterBand(1)
GPPdata = GPPband.ReadAsArray()
GPParray = GPPdata.astype(float)

# 目标序列
last = np.array([1,1,0])
first = np.array([0,1,1])
one = np.array([0,1,0])

last1 = last.reshape((3, 1))
first1 = first.reshape((3, 1))
one1 = one.reshape((3, 1))

# result = np.zeros((GPProws, 0)) # 使用列表存储每列的结果
columns = []
window_size = len(last)
GPP_Drought_befor = 0
GPPmean = 0
start_time = 0
end_time = 0

# for j in tqdm(range(start_column, end_column)):
#     for i in range(GPProws-2):
column = band.ReadAsArray(start_column, 0, 1, GPProws)
# print(column)
# column = np.zeros((20, 0))
result = np.zeros((GPProws, end_column - start_column))

def process_column(j):
    column = np.zeros((GPProws,))  # 为每次迭代初始化列数组
    GPPmean = 0
    for i in range(GPProws - 2):
        if np.sum(array[:, j]) == 0:
            continue
        window = array[i:i + window_size, j:j + 1]
        if np.all(window == one1) and i > 2:
            GPP_Drought_befor = GPParray[[i, i - 1, i - 2], j:j + 1]
            GPPmean = np.mean(GPP_Drought_befor)
            column[i + 1] = GPPmean
        if np.all(window == first1):
            start_time = i + 1
            GPP_Drought_befor = GPParray[[i, i - 1, i - 2], j:j + 1]
            GPPmean = np.mean(GPP_Drought_befor)
        if np.all(window == last1):
            end_time = i + 1
            column[i + 1] = GPPmean

    return column

# 使用并行计算处理循环迭代
results = Parallel(n_jobs=-1)(delayed(process_column)(j) for j in tqdm(range(start_column, end_column)))

# 转换结果为数组
result = np.column_stack(results)

output_tif_path = r"G:\粗分辨率补充\4旱前均值\kNDVI旱前均值12月.tif"
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(
    output_tif_path,
    end_column - start_column,
    GPProws,
    1,
    gdal.GDT_Float32,
)

output_dataset.SetProjection(filename.GetProjection())
output_dataset.SetGeoTransform(filename.GetGeoTransform())
output_dataset.GetRasterBand(1).WriteArray(result)
output_dataset.FlushCache()
output_dataset = None

