from PIL import Image
from pylab import *
import cv2
import os


def point(img_path):
    Image.MAX_IMAGE_PIXELS = 100000000000
    # im = array(Image.open('best_result.png'))
    im = array(Image.open(img_path))
    imshow(im)
    print('Please click 2 points')
    x = ginput(2)
    # show()
    return x


#output_path = 'E:/UltraSPic/frame/'
output_path = 'E:/UltraSPic/jpg/'

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'X: {x}, Y: {y}')
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'({x},{y})', (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)


if __name__ == '__main__':
    files = [f for f in os.listdir(output_path) if f.endswith('.png')]
    files.sort(key=lambda x: int(x[:-4]))  # 假设文件名是数字命名的

    for file in files:
        img_path = os.path.join(output_path, file)
        img = cv2.imread(img_path)

        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)

        print("Press any key to go to the next image or 'q' to quit")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     x = point('cut_orchard2.png')
#     print('you clicked:', x)


