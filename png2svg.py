import os
import cv2
from shapely.geometry import Polygon
from shapely.geometry import mapping
from fiona import collection
# 创建输出文件夹
output_folder = '/folder'
os.makedirs(output_folder, exist_ok=True)
# 遍历文件夹中的所有.png文件
for filename in os.listdir('/'):
    if filename.endswith('.png'):
        # 读取二值化图像
        image_path = os.path.join('/', filename)
        binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 找到轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建.shp文件
        shapefile_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.shp')
        with collection(shapefile_path, "w", "ESRI Shapefile", schema={"geometry": "Polygon"}) as output:
            for contour in contours:
                # 检查轮廓是否至少包含 3 个点
                if len(contour) >= 3:
                    # 将轮廓点转换为Shapely Polygon
                    polygon = Polygon(contour.reshape(-1, 2))

                    # 将Shapely Polygon转换为GeoJSON格式
                    feature = {
                        "type": "Feature",
                        "properties": {},
                        "geometry": mapping(polygon)
                    }

                    # 写入到.shp文件中
                    output.write(feature)

        print(f'已创建矢量图像文件：{shapefile_path}')

print(f'所有矢量图像已创建完毕。')



