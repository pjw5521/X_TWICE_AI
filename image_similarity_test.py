import torch
import torch.nn  as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from PIL import Image


# image similarity 

class Image_Similarity():

    def __init__(self) -> None:
        self.model =  models.vgg16(pretrained=True) # 사전에 훈련된 모델
        self.New_model = nn.Sequential(*(list(self.model.children())[0:1]))

    # model의 결과를 numpy로
    def forward(self, img):

        result = self.New_model(img.unsqueeze(0))
        result = result.view(-1, 512 * 7 * 7).cpu()
        result = result.detach().numpy()

        return result
    
    # compute image similarity
    def Compute_sim(img1_vec, img2_vec):
        return np.dot(img1_vec, img2_vec) / (np.norm(img1_vec) * np.norm(img2_vec))

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

# main 함수
if __name__ == '__main__':

    # GPU 설정
    print(torch.cuda.is_available())
    print(torch.cuda.device_count()) # 2개 

    cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(cuda)

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

    # data load 불러오기
    imag_file = Image_File()
    img_train = imag_file.image_file(path_train, name_train)

    # data transform
    transform = transforms.Compose





            

