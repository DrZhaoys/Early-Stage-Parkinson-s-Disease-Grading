from img_point import point
import cv2
import numpy as np
import os


def cut_pic(img_path, point_list):
    [(x1, y1), (x2, y2)] = point_list
    img = cv2.imread(img_path)
    # img = np.array(img)
    # 新图片
    print(img.shape)
    # 此处注意opencv的顺序是先h，后w，然后c。
    img_new = img[y1:y2, x1:x2, :]
    #cv2.imwrite("./crop/", img_new)
    return img_new


def save_image(image, output_path):
    image.save(output_path)


def read_file(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            processed_image = cut_pic(input_file_path, point_two_new)
            #save_image(processed_image, output_file_path)
            cv2.imwrite(os.path.join(output_folder, filename), processed_image)


if __name__ == '__main__':
    #dir_path = 'E:/UltraSPic/jpg'

    #img_path = 'E:/USP/1/22.jpg'

    # 得到两个对角点的坐标
    #point_two = point(img_path)
    #point_two_new = [(709, 266), (1481, 866)] #img裁剪坐标
    point_two_new = [(142, 94), (723, 541)] #frame裁剪坐标
    # # 清空列表
    # point_two.clear()
    # for i in point_two:
    #    i_new = list(map(int, i))
    #    point_two_new.append(i_new)
    # print(point_two_new)
    #cut_pic(img_path, point_two_new)

    # input_folder = 'E:/UltraSPic/frame'  # 输入文件夹路径
    # output_folder = 'E:/UltraSPic/frame_crop'  # 输出文件夹路径
    #input_folder = 'E:/UltraSPic/jpg'  # 输入文件夹路径
    #output_folder = 'E:/UltraSPic/jpg_crop'  # 输出文件夹路径
    #input_folder = 'E:/UltraSPic/Origin_data_all/data3/frame'
    #output_folder = 'E:/UltraSPic/Origin_data_all/data3/frame_crop'
    input_folder = 'E:/UltraSPic/Origin_data_all/data2/frame'
    output_folder = 'E:/UltraSPic/Origin_data_all/data2/frame_crop'
    read_file(input_folder, output_folder)


