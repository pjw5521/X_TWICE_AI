import os
from matplotlib.image import imread
import numpy as np
from numpy import dot
from numpy.linalg import norm
import re
from PIL import Image
import torchvision.transforms as transforms
import torch
from model_2 import Image_Similarity

# image similarity class
class Image_Similarity_class():

    def __init__(self, model_path) -> None:
        self.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.New_model = Image_Similarity()
        self.New_model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        self.New_model.to(self.cuda)

    # model의 결과를 numpy로
    def forward(self, img):

        result = self.New_model(img)
        result = result.view(-1, 512 * 2 * 1).cpu()
        result = result.squeeze(0).detach().numpy()

        return result
    
    # compute image similarity
    def Compute_sim(self, img1_vec, img2_vec):
        return dot(img1_vec, img2_vec) / (norm(img1_vec) * norm(img2_vec))

    
# Image File
class Image_File():

    def __init__(self) -> None:
        pass    
    # image data load
    def image_file(self, path, filename, num_list):
        # result 
        img_file = []
        file_list = os.listdir(path)

        for num in num_list:
            img_file.append(Image.open(path + '/' + filename + str(num) + '.jpg'))
        
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
            result_image.append(img.to(cuda))

        return result_image


if __name__ == '__main__':

    path_train = '../test/train'
    path_mirror = '../test/mirror'
    path_dark = '../test/dark'

    #이미지 파일 이름
    name_train = "train"
    name_rotate = "rotate"
    name_mirror = "mirror"
    name_bright = "bright"
    name_dark = "dark"

    num_list = [5, 14, 19, 87, 3] # 각각 mirror & dark image detect 잘함

    # setting
    image_file = Image_File()

    # image_list
    train_images = image_file.image_file(path_train, name_train, num_list)
    mirror_images = image_file.image_file(path_mirror, name_mirror, num_list)
    dark_images = image_file.image_file(path_dark, name_dark, num_list)

    train_images = image_file.image_transform(train_images)
    mirror_images = image_file.image_transform(mirror_images)
    dark_images = image_file.image_transform(dark_images)

    ## similarity calculate

    img_sim = Image_Similarity_class('../My_model/cosine_Vgg_3.pt')

    vs_result = img_sim.forward(train_images[4])
    print("num: ", num_list[4])
    print("3번 vector: ", vs_result)

    for i in range(len(train_images) - 1):

        result_train = img_sim.forward(train_images[i])
        result_mirror = img_sim.forward(mirror_images[i])
        result_dark = img_sim.forward(dark_images[i])

        sim1 = img_sim.Compute_sim(result_train, result_mirror)
        sim2 = img_sim.Compute_sim(result_train, result_dark)
        sim3 = img_sim.Compute_sim(result_train, vs_result)

        print("sim1: {}, sim2: {}, sim3: {}", sim1, sim2, sim3)
    
    ## 안 맞는 이미지 찾기
 
    


    






