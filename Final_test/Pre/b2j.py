import os
import cv2

#bmp_dir = r'E:\UltraSPic'
#jpg_dir = r'E:\UltraSPic\jpg'

bmp_dir = r'E:\UltraSPic\Origin_data_all\data2\bmp'
jpg_dir = r'E:\UltraSPic\Origin_data_all\data2\jpg'

filelists = os.listdir(bmp_dir)

for i, file in enumerate(filelists):
    img = cv2.imread(os.path.join(bmp_dir, file), -1)
    newName = file.replace('.bmp', '.jpg')
    cv2.imwrite(os.path.join(jpg_dir, newName), img)
    print('第%d张图：%s' % (i+1, newName))

