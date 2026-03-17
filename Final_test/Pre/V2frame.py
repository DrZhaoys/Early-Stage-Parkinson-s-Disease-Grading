import cv2
import os
#video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-28-57_1.avi'

#video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-29-32_1.avi'
#video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-29-53_1.avi'
#video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-29-57_1.avi'
#video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-30-02_1.avi'
#video_path = 'E:/学习/文献阅读/SCI-HUB/PD/PD超声图像数据/100frame/2023-11-21-12-14-41_2023-11-07-12-31-31_1.avi'
#video_folder_path = 'E:/UltraSPic/all_video/'
#output_path = 'E:/UltraSPic/frame/'
video_folder_path = 'E:/UltraSPic/Origin_data_all/data2/video/'
output_path = 'E:/UltraSPic/Origin_data_all/data2/frame/'
# interval = 2  # 间隔interval帧取一张

files = os.listdir(output_path)
#files.sort()  # 可以确保文件是按照名称排序的
files.sort(key=lambda x: int(x[:-4]))
last_filename = files[-1] if files else None
last_num_int = int(os.path.splitext(last_filename)[0]) if last_filename else 0
#last_num, extension = os.path.splitext(last_filename)
#last_num_int = int(last_num)
#last_file_number = 98
# 创建新文件名
#new_filename = f"{last_file_number + 1}.txt"
#new_file_path = os.path.join(folder_path, new_filename)


def extract_frames_from_video(video_path, last_num_int, start_num):
    num = start_num
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # 帧率
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  # 总的帧数
    print(f"Processing {video_path}: fps={int(fps)}, frames={int(frames)}")

    while video.isOpened():
        is_read, frame = video.read()
        if is_read:
            file_name = f"{last_num_int + num:05d}"  # 使用零填充格式化编号
            cv2.imwrite(os.path.join(output_path, f"{file_name}.png"), frame)  # 输出图片
            cv2.waitKey(1)
            num += 1
        else:
            break

    video.release()
    return num - start_num


if __name__ == '__main__':
    video_files = [f for f in os.listdir(video_folder_path) if f.endswith(('.avi', '.mp4', '.mkv'))]
    video_files.sort()

    for video_file in video_files:
        video_path = os.path.join(video_folder_path, video_file)
        frames_extracted = extract_frames_from_video(video_path, last_num_int, 1)
        last_num_int += frames_extracted  # 更新编号以确保连续编号

    print("所有视频帧提取完成")
#    num = 1
#    video = cv2.VideoCapture(video_path)
#    fps = video.get(cv2.CAP_PROP_FPS)  # 帧率
#   frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  # 总的帧数
#     print("fps=", int(fps), "frames=", int(frames))
#     while video.isOpened():
#         is_read, frame = video.read()
#         if is_read:
#             file_name = f"{last_num_int + num}"
#             cv2.imwrite(output_path + str(file_name) + '.png', frame)  # 输出图片
#             cv2.waitKey(1)
#             num += 1
#         else:
#             break
#
#     video.release()
#     print("视频帧提取完成")


