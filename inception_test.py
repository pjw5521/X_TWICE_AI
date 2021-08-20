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
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# image similarity class
class Image_Similarity():

    def __init__(self) -> None:
        cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model =  torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True) # 사전에 훈련된 모델
        self.New_model = nn.Sequential(*list(self.model.children())[0:-1]).to(cuda)
        #self.New_model = self.model
        print(self.New_model)

    # model의 결과를 numpy로
    def forward(self, img):

        with torch.no_grad():
            img = img.unsqueeze(0)
            print(img.size())
            result = self.New_model(img)
            print(result.size())
            result = result.view(-1, 512 * 7 * 7).cpu()
            result = result.squeeze(0).detach().numpy()

        return result
    
    # compute image similarity
    def Compute_sim(self, img1_vec, img2_vec):
        return dot(img1_vec, img2_vec) / (norm(img1_vec) * norm(img2_vec))

    
    # result method
    def Resutl_Top_4(self, train_f,  rotate_f, mirror_f, dark_f, bright_f):

        #debug 
        print(len(train_f))
        print(len(rotate_f))
        print(len(mirror_f))
        print(len(dark_f))
        print(len(bright_f))

        ## Result 값 
        Result_total = [] # 해당 이미지가 몇 개가 일치하는가?
        Result_value = [] # 일치하는 값들의 대략적인 Similarity 값 저장
        Result_index = [] # 일치하는 이미지의 index
        Result_Not_index = [] # 일치하지않는 이미지는 무엇인가?
        Result_Not_value = [] # 일치하지않는 이미지의 대략적인 Similarity 값?

        # similarity
        for t_idx, train_vec in enumerate(train_f):

            # print("index " + str(t_idx))
            
            # image 하나의 결과
            result = []
            result_value = []
            result_index = []
            result_Not_index = []
            result_Not_value = [] 
            
            result_r = []
            result_m = []
            result_d = []
            result_b = []

            # rotate similarity
            for rotate_vec in rotate_f: # 10개
                result_r.append(self.Compute_sim(train_vec, rotate_vec))
            result.append(result_r)
            
            for mirror_vec in mirror_f:
                result_m.append(self.Compute_sim(train_vec, mirror_vec))
            result.append(result_m) 
            
            for dark_vec in dark_f:
                result_d.append(self.Compute_sim(train_vec, dark_vec))
            result.append(result_d)
            
            for bright_vec in bright_f:
                result_b.append(self.Compute_sim(train_vec, bright_vec))          
            result.append(result_b)

            result = np.array(result)

            ## accuracy - debug
            print("max")
            print(result)
            #print(np.max(result))
            #print(np.argmax(result)) # index가 일렬로 나오는 

            count = 0

            for i in range(0, 4):

                # result의 max 값의 index
                index = np.argmax(result)
                print(np.max(result))
                print("index " +  str(int(index/100)))
                if t_idx == int(index % 100):
                    count += 1
                    # 해당 max값 & index를 저장
                    result_value.append(np.max(result))
                    result_index.append(int(index/100))
                    # result_index.append(t_idx)
                    result[int(index/100)][int(index % 100)] = 0
                # 일치하지않을 경우
                else:
                    # debug
                    print("else")
                    print(np.max(result))
                    print(int(index/100)) 

                    result_Not_value.append(np.max(result))
                    result_Not_index.append(int(index/100))
                    result[int(index/100)][int(index % 100)] = 0
            

            # 최종 Result에 append
            Result_total.append(count)
            Result_value.append(result_value)
            Result_index.append(result_index)
            Result_Not_index.append(result_Not_index)
            Result_Not_value.append(result_Not_value)
            
        # debug
        print("accuracy")    
        print(Result_total)
        print("accuracy value")
        print(Result_value)
        print("accuracy index")
        print(Result_index)
        print("Not consistent: ")
        print("index")
        print(Result_Not_index)
        print("Not_value")
        print(Result_Not_value)

        # average
        total_first = 0
        total_second = 0
        total_third = 0
        total_forth = 0

        for t_value in Result_value:
            for idx, value in t_value:
                pass

        # matplot show
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')

        for i in range(1, len(Result_value) + 1):
            for j in range(0, len(Result_value[i - 1])):
                ax1.scatter(i, Result_index[i - 1][j], Result_value[i - 1][j], c = Result_value[i - 1][j], cmap = 'jet')
        
        ax1.set_xlabel('image')
        ax1.set_ylabel('index')
        ax1.set_zlabel('similarity')
        ax1.set_title('Consistent image')
        ax1.view_init(40, -60)
        ax1.invert_xaxis()

        
        ax2 = fig.add_subplot(1, 3, 2, projection = '3d')
        
        for i in range(1, len(Result_Not_value) + 1):
            for j in range(0, len(Result_Not_value[i - 1])):
                ax2.scatter(i, Result_Not_index[i - 1][j], Result_Not_value[i - 1][j], c = Result_Not_value[i - 1][j], cmap = 'jet')
        
        ax2.set_xlabel('image')
        ax2.set_ylabel('index')
        ax2.set_zlabel('similarity')
        ax2.set_title('Not Consistent image')
        ax2.view_init(40, -60)
        ax2.invert_xaxis()
        
        plt.show()

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
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # cuda
        cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        result_image = []

        for image in img_list:
            # print(image)
            img = transform(image)
            result_image.append(img.to(cuda))

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

    ##debug
    print(img_trans_train[0].shape)
    #print(img_trans_rotate)
    #print(img_trans_mirror)
    #print(img_trans_dark)
    #print(img_trans_bright)
    
    
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
    for i in range(0, 100):
        print(str(i) + "image")

        img_result_train.append(img_sim.forward(img_trans_train[i]))
        img_result_rotate.append(img_sim.forward(img_trans_rotate[i]))
        img_result_mirror.append(img_sim.forward(img_trans_mirror[i]))
        img_result_dark.append(img_sim.forward(img_trans_dark[i]))
        img_result_bright.append(img_sim.forward(img_trans_bright[i]))
    
    ''' debug
    print(len(img_result_train))
    print(len(img_result_rotate))
    print(len(img_result_mirror))
    print(len(img_result_dark))
    print(len(img_result_bright))
    '''

    # result method
    img_sim.Resutl_Top_4(img_result_train, img_result_rotate, img_result_mirror, img_result_dark, img_result_bright)




    


   






            

