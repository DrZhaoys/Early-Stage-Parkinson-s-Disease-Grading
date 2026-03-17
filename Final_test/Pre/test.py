import cv2
import os
#video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-28-57_1.avi'

video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-29-32_1.avi'
output_path = 'E:/UltraSPic/frame/'


numbers = os.path.basename(output_path)
filename, extension = os.path.splitext(os.path.basename(output_path))

print(numbers)
print(filename)
