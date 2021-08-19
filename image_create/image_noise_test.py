import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

# transform된 이미지 저장
def image_save(path, filename, img_list):

    print("image_save " + filename)

    for i in range(1, len(img_list) + 1):
        img = cv2.imread(path + '/' + filename + str(i) + '.jpg')
        #print(type(img))

        if str(type(img)) == "<class 'NoneType'>":
            #print("save")
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

# image rotate 오른쪽 90도
def rotate_img(img_list):

    img_rot = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in img_list ] # 시계방향으로 90도 회전

    return img_rot

# 45도 회전
def rotate_img_1(img_list):

    img_rot = [imutils.rotate(img, 45) for img in img_list ] # 시계방향으로 90도 회전

    return img_rot

# 135도 회전
def rotate_img_2(img_list):

    img_rot = [imutils.rotate(img, 90 + 45) for img in img_list]

    return img_rot

# 180도 회전
def rotate_img_3(img_list):

    img_rot = [cv2.rotate(img, cv2.ROTATE_180) for img in img_list]

    return img_rot

# 225도 회전

def rotate_img_4(img_list):

    img_rot = [imutils.rotate(img, 225) for img in img_list]
    return img_rot

# 270도 회전
def rotate_img_5(img_list):

    img_rot = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in img_list] # 반시계방향 90도 =  시계 방향 270도
    return img_rot

# 315도 회전
def rotate_img_6(img_list):

    img_rot = [imutils.rotate(img, 270 + 45) for img in img_list]
    return img_rot

# 상하반전 image filp
def flip_img(img_list):
    img_flip_ud = [cv2.flip(img, 0) for img in img_list]

    return img_flip_ud

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
    path_train = '../data/train'
    path_rotate = '../data/rotate'
    path_rotate1 = '../data/rotate1'
    path_rotate2 = '../data/rotate2'
    path_rotate3 = '../data/rotate3'
    path_rotate4 = '../data/rotate4'
    path_rotate5 = '../data/rotate5'
    path_rotate6 = '../data/rotate6'
    path_mirror = '../data/mirror'
    path_bright = '../data/bright'
    path_dark = '../data/dark'
    path_flip = '../data/filp'

    img_train = image_file(path_train, 100)
    print(len(img_train[0]))
    
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
    # image rotate1
    img_rot_1 = rotate_img_1(img_train)
    print(len(img_rot_1))

    ## debug
    plt.imshow(img_rot_1[0])
    plt.show()

    # image rotate
    img_rot_2 = rotate_img_2(img_train)
    print(len(img_rot_2))

    ## debug
    plt.imshow(img_rot_2[0])
    plt.show()

    # image rotate
    img_rot_3 = rotate_img_3(img_train)
    print(len(img_rot_3))

    ## debug
    plt.imshow(img_rot_3[0])
    plt.show()

    # image rotate
    img_rot_4 = rotate_img_4(img_train)
    print(len(img_rot_4))

    ## debug
    plt.imshow(img_rot_4[0])
    plt.show()

    # image rotate
    img_rot_5 = rotate_img_5(img_train)
    print(len(img_rot_5))

    ## debug
    plt.imshow(img_rot_5[0])
    plt.show()

    # image rotate
    img_rot_6 = rotate_img_6(img_train)
    print(len(img_rot_6))

    ## debug
    plt.imshow(img_rot_6[0])
    plt.show()

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

    img_flip = flip_img(img_train)
    print(len(img_flip))

    # debug
    plt.imshow(img_flip[0])
    plt.show()

    
    print("image save start")
    #image save
    name_rot = "rotate"
    image_save(path_rotate, name_rot , img_rot)

    name_rot_1 = "rotate1_"
    image_save(path_rotate1, name_rot_1 , img_rot_1)

    name_rot_2 = "rotate2_"
    image_save(path_rotate2, name_rot_2 , img_rot_2)

    name_rot_3 = "rotate3_"
    image_save(path_rotate3, name_rot_3 , img_rot_3)
    
    name_rot_4 = "rotate4_"
    image_save(path_rotate4, name_rot_4 , img_rot_4)

    name_rot_5 = "rotate5_"
    image_save(path_rotate5, name_rot_5 , img_rot_5)

    name_rot_6 = "rotate6_"
    image_save(path_rotate6, name_rot_6 , img_rot_6)

    name_mirror = "mirror"
    image_save(path_mirror, name_mirror, img_mirror)

    name_bright = "bright"
    image_save(path_bright, name_bright, img_bright)

    name_dark = "dark"
    image_save(path_dark, name_dark, img_dark)

    name_flip = "filp"
    image_save(path_flip, name_flip, img_flip)

    print("image save finish")
    
    
    



    


