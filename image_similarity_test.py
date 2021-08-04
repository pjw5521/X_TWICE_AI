import torch
import torch.nn  as nn
from torch.utils.data.dataloader import T_co
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image

# image similarity class
class Image_Similarity():

    def __init__(self) -> None:
        cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model =  models.vgg16(pretrained=True) # 사전에 훈련된 모델
        self.New_model = nn.Sequential(*(list(self.model.children())[0:1]))

    # model의 결과를 numpy로
    def forward(self, img):

        result = self.New_model(img.unsqueeze(0))
        result = result.view(-1, 512 * 7 * 7).cpu()
        result = result.squeeze(0).detach().numpy()

        return result
    
    # compute image similarity
    def Compute_sim(self, img1_vec, img2_vec):
        return dot(img1_vec, img2_vec) / (norm(img1_vec) * norm(img2_vec))

    
    # result method
    def Resutl_Top_3(self, train_f, )

# Image File
class Image_File():

    def __init__(self) -> None:
        pass    
    # image data load
    def image_file(self, path, filename):
        # result 
        img_file = []
        file_list = os.listdir(path)

        for i in range(1, len(file_list) + 1):
            img_file.append(Image.open(path + '/' + filename + str(i) + '.jpg'))
        
        return img_file

    # image file path list
    def make_file_list(self, path, filename):

        train_img_list = list()

        for i in range(1, 101):
            img_path = path + '/' + filename + str(i) + '.jpg'
            train_img_list.append(img_path)

        return train_img_list

    #image transform & cuda data로 바꿔주기
    def image_transform(self, img_list):

        # print(img_list) - debug

        # data transform
        transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # cuda
        cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        result_image = []

        for image in img_list:
            # print(image)
            img = transform(image)
            result_image.append(img)

        return result_image
    
# main 함수
if __name__ == '__main__':

    # GPU 설정
    print(torch.cuda.is_available()) # True
    print(torch.cuda.device_count()) # 2개 

    # cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(cuda)

    # 이미지 파일 경로
    path_train = './data/train'
    path_rotate = './data/rotate'
    path_mirror = './data/mirror'
    path_bright = './data/bright'
    path_dark = './data/dark'

    #이미지 파일 이름
    name_train = "train"
    name_rotate = "rotate"
    name_mirror = "mirror"
    name_bright = "bright"
    name_dark = "dark"

    # all data load 불러오기
    image_file = Image_File()
    img_train = image_file.image_file(path_train, name_train)
    img_rotate = image_file.image_file(path_rotate, name_rotate)
    img_mirror = image_file.image_file(path_mirror, name_mirror)
    img_dark = image_file.image_file(path_dark, name_dark)
    img_bright = image_file.image_file(path_bright, name_bright)

    # transform data
    img_trans_train = image_file.image_transform(img_train)
    img_trans_rotate = image_file.image_transform(img_rotate)
    img_trans_mirror = image_file.image_transform(img_mirror)
    img_trans_dark = image_file.image_transform(img_dark)
    img_trans_bright = image_file.image_transform(img_bright)

    ''' debug
    print(img_trans_train)
    print(img_trans_rotate)
    print(img_trans_mirror)
    print(img_trans_dark)
    print(img_trans_bright)
    '''
    
    # Image Smilarity 계산
    img_sim = Image_Similarity()

    # image feature vector
    img_result_train = []
    img_result_rotate = []
    img_result_mirror = []
    img_result_dark = []
    img_result_bright = []
    
    print("image sim 계산")

    #  image feature vector: 우선 cpu로 10개 이미지 해보기  
    for i in range(0, 10):
        print(str(i) + "image")
        img_result_train.append(img_sim.forward(img_trans_train[i]))
        img_result_rotate.append(img_sim.forward(img_trans_rotate[i]))
        img_result_mirror.append(img_sim.forward(img_trans_mirror[i]))
        img_result_dark.append(img_sim.forward(img_trans_dark[i]))
        img_result_bright.append(img_sim.forward(img_trans_bright[i]))
    

    # Result method

    # gitub에 올라가나용?



    


   






            

