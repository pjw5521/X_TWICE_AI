import os
import cv2
from matplotlib import pyplot as plt

def image_save():
    return 0


def image_file(path, count):
    
    # return할 cv2 img list
    cv_img_list = []

    for i in range(1, count + 1):

        # cv2로 이미지 불러오기 
        img = cv2.imread(path + '/train' + str(i) +'.jpg')
        # debug image show
        plt.imshow(img)
        plt.show()
        #cv2.imshow("image", img)
        cv_img_list.append(img)
    
    return cv_img_list


if __name__ == '__main__':

    # 파일 경로
    path_train = './data/train'
    path_rotate = './data/rotate'
    path_bright = './data/bright'

    img_train = image_file(path_train, 100)
    print(img_train)

    


