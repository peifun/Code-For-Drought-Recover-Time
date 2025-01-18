import statsmodels.api as sm
import numpy as np
from osgeo import gdal
from tqdm import tqdm

# 打开.tif文件
dataset = gdal.Open(r"D:\城镇化与植被数据\滞后期\平滑\SPEI01.tif")

if dataset is None:
    print("无法打开文件")
    exit()

# 获取.tif文件的列数和行数
cols = dataset.RasterXSize
rows = dataset.RasterYSize
band = dataset.GetRasterBand(1)  # GDAL的波段索引从1开始
data = band.ReadAsArray()
print(data.shape)

zero_array = [0] * rows

projection = dataset.GetProjection()  # 投影
geotrans = dataset.GetGeoTransform()  # 几何信息

result_all = []
# 读取每一列数据并执行操作
for col in tqdm(range(cols)):
    # 读取当前列的数据
    column_data = data[:,col]
    # if np.sum(column_data)!=0:
    #     print(column_data)
    if column_data[0] == 0 and column_data[1]== 0 and column_data[2]== 0:
        result_all.append(zero_array)
        continue

    # 在这里执行你的操作
    # 去季节化
    result = sm.tsa.seasonal_decompose(column_data, model='additive', period=12)
    resultS = result.resid
    # 加一个去趋势以填补上一步中前六后六的空值
    time_index = np.arange(len(column_data))
    X = sm.add_constant(time_index)
    model = sm.OLS(column_data, X)
    resultQS = model.fit()
    # 提取去趋势化后的数据
    add_data = resultQS.resid
    # print(detrended_data)
    # result_all.append(detrended_data)
    resultS[0:6] = add_data[0:6]
    resultS[-6:] = add_data[-6:]
    last = resultS
    result_all.append(last)
# # 整合数据并输出(去趋势化)
result_all_new = np.reshape(result_all,(cols,rows))
result_all_new = np.transpose(result_all_new,(1,0))
print(result_all_new.shape)

driver = gdal.GetDriverByName("GTiff")
dataset = driver.Create(r"D:\城镇化与植被数据\滞后期\去趋势去季节\SPEI03.tif", cols, rows, 1, gdal.GDT_Float32)
dataset.SetGeoTransform(geotrans)
dataset.SetProjection(projection)
dataset.GetRasterBand(1).WriteArray(result_all_new)
del dataset

# 关闭文件
dataset = None

