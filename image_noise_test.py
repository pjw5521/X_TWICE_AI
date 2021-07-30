import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

# transform된 이미지 저장
def image_save(path, filename, img_list):

    print("image_save " + filename)

    for i in range(1, len(img_list) + 1):
        img = cv2.imread(path + '/' + filename + str(i) + '.jpg')
        print(type(img))

        if str(type(img)) == "<class 'NoneType'>":
            cv2.imwrite(path + '/' + filename + str(i) + '.jpg', img_list[i - 1])
        else: 
            print("already exist")
            break

    print("success")

# image file에서 불러오기
def image_file(path, count):
    
    # return할 cv2 img list
    cv_img_list = []

    for i in range(1, count + 1):

        # cv2로 이미지 불러오기 
        img = cv2.imread(path + '/train' + str(i) +'.jpg')
        
        # debug image show
        #plt.imshow(img)
        #plt.show()
        
        #cv2.imshow("image", img)
        cv_img_list.append(img)
    
    return cv_img_list    

# image rotate
def rotate_img(img_list):

    img_rot = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in img_list ] # 시계방향으로 90도 회전

    return img_rot

# image mirror
def mirror_img(img_list):
    img_mirror = [cv2.flip(img, 1) for img in img_list]
    return img_mirror

# image bright
def dark_img(img_list):
    # 50정도 어둡게
    dark = [np.ones(img.shape,dtype="uint8") * 50 for img in img_list]
    
    img_dark = []

    for i in range(0, len(img_list)):
        img_dark.append(cv2.subtract(img_list[i], dark[i]))

    return img_dark    

# image dark
def bright_img(img_list):

    bright = [ np.ones(img.shape,dtype="uint8") * 100 for img in img_list]
    img_bright = []

    # 100 밝기 높이기
    for i in range(0, len(img_list)):
        img_bright.append(cv2.add(img_list[i], bright[i]))

    return img_bright

if __name__ == '__main__':

    # 파일 경로
    path_train = './data/train'
    path_rotate = './data/rotate'
    path_mirror = './data/mirror'
    path_bright = './data/bright'
    path_dark = './data/dark'

    img_train = image_file(path_train, 100)
    print(len(img_train))
    
    # debug
    plt.imshow(img_train[0])
    plt.show()
    
    # image rotate
    img_rot = rotate_img(img_train)
    print(len(img_rot))
    
    '''debug
    plt.imshow(img_rot[0])
    plt.show()
    '''

    # image mirror
    img_mirror = mirror_img(img_train)
    print(len(img_mirror))

    ''' debug
    plt.imshow(img_mirror[0])
    plt.show()
    '''

    #image bright
    img_bright = bright_img(img_train)
    print(len(img_bright))

    '''debug
    plt.imshow(img_bright[0])
    plt.show()
    '''

    # image dark
    img_dark = dark_img(img_train)
    print(len(img_dark))

    # debug
    plt.imshow(img_dark[0])
    plt.show()
    
    print("image save start")
    #image save
    name_rot = "rotate"
    image_save(path_rotate, name_rot , img_rot)

    name_mirror = "mirror"
    image_save(path_mirror, name_mirror, img_mirror)

    name_bright = "bright"
    image_save(path_bright, name_bright, img_bright)

    name_dark = "dark"
    image_save(path_dark, name_dark, img_dark)

    print("image save finish")
    
    



    


