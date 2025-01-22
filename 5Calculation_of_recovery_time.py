import numpy as np
from osgeo import gdal
from tqdm import tqdm

# 打开判断完的文件
dataset = gdal.Open(r"G:\粗分辨率补充\4旱前均值\kNDVI旱前均值12月.tif")

# 打开没判断文件
dataset2 = gdal.Open(r"G:\粗分辨率补充\2去趋势\kNDVI去趋势.tif")

# 获取SPEI文件的列数和行数
cols = dataset.RasterXSize
rows = dataset.RasterYSize
band = dataset.GetRasterBand(1)  # GDAL的波段索引从1开始
data = band.ReadAsArray()

# 获取GPP文件的列数和行数
band2 = dataset2.GetRasterBand(1)  # GDAL的波段索引从1开始
data2 = band2.ReadAsArray()

projection = dataset2.GetProjection()  # 投影
geotrans = dataset2.GetGeoTransform()  # 几何信息

recover_time_all = []
recover_final = []
print("cols",cols)
print("rows",rows)


for col in tqdm(range(cols)):
    # 读取当前列的数据
    column_data = data[:, col]  # 读取整列数据
    column_data2 = data2[:, col]
    drought = 0
    recover_time = -1
    if np.sum(column_data) == 0:
        recover_final.append(0)
        continue
    for i in range(len(column_data)):
        if i-drought > recover_time and column_data[i] != 0:
            Mean = column_data[i]
            drought = i
            j = drought
            for j in range(j+1,len(column_data2)):
                if column_data2[j] > Mean:
                    recover_time = j - drought
                    recover_time_all.append(recover_time)
                    break
    if len(recover_time_all) != 0:
        Recover = sum(recover_time_all) / len(recover_time_all)
    else:
        Recover = sum(recover_time_all)
    recover_final.append(Recover)
    recover_time_all = []
print(len(recover_final))
recover_tif = np.reshape(recover_final,(88,137))#行列
driver = gdal.GetDriverByName("GTiff")
dataset = driver.Create(r"G:\粗分辨率补充\6恢复时间\恢复时间12月.tif", 137, 88, 1, gdal.GDT_Int16)#列行
dataset.SetGeoTransform(geotrans)
dataset.SetProjection(projection)
dataset.GetRasterBand(1).WriteArray(recover_tif)
del dataset



