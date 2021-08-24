### data 전처리
import os
from matplotlib.image import imread
import numpy as np
import re
from PIL import Image
import torchvision.transforms as transforms
import torch

class PreProcessing:

    # data 
    images_train = np.array([])
    images_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self, data_src, ratio):
        self.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_scr = data_src
        print("image data load 중~")
        self.images_train, self.images_test, self.labels_train, self.labels_test = self.preprocessing(ratio) 
        self.unique_train_label = np.unique(self.labels_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in 
                                        self.unique_train_label  }

        # debug
        print('self.images_train : ', len(self.images_train))
        print('self.images_test : ', len(self.images_test))
        print('self.labels_train : ', len(self.labels_train))
        print('self.labels_test : ', len(self.labels_test))
        print('self.unique_train_label : ', self.unique_train_label)
        print('self.map_train_label_indices : ', self.map_train_label_indices)

    # normalize
    def Normalize(self, x):

        meanRGB = []
        stdRGB = []
        
        array_x = np.array(x)
        #print(array_x.shape)
        #print(np.mean(array_x[:,:,0]) / 255.)

        meanRGB.append(np.mean(array_x[:,:,0]) / 255.)
        meanRGB.append(np.mean(array_x[:,:,1]) / 255.)
        meanRGB.append(np.mean(array_x[:,:,2]) / 255.)

        stdRGB.append(np.std(array_x[:,:,0]) / 255.)
        stdRGB.append(np.std(array_x[:,:,1]) / 255.)
        stdRGB.append(np.std(array_x[:,:,2]) / 255.)

        ''' debug
        print('meanRGB :', meanRGB)
        print('stdRGB :', stdRGB)
        '''
        return meanRGB, stdRGB

    # tensor 형태로 GPU로
    def Transfrom(self, img):

        img_meanRGB, img_stdRGB = self.Normalize(img)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(img_meanRGB, img_stdRGB)
        ])
        
        return transform(img).to(self.cuda)
    
    # image data read
    def read_data(self):
        # 전체 data set 
        images = []
        labels = []

        for dir in os.listdir(self.data_scr):
            #print(dir)
            try:
                for pic in os.listdir(os.path.join(self.data_scr, dir)):
                    
                    # image data
                    img = Image.open(self.data_scr + dir + '/' + pic)
                    images.append(img)
                    # label create
                    # print('jpg file name; ', pic[:-4])
                    num = re.findall('\d+', pic[:-4])
                    if len(num) == 1:
                        labels.append(int(num[0]))
                    elif len(num) == 2:
                        labels.append(int(num[1]))
                    
            except Exception as e:
                print('image data load에 실패. :', dir)
                print('Exception Message: ', e)
        
        ''' debug
        print(images[0])
        print(len(labels))
        
        , pos_images, neg_images
        for i in range(1, 100 + 1):
            print(labels.count(i))
        '''
        return images, labels
    
    # data 전처리
    def preprocessing(self, ratio):
        
        # 각 images -> PIL type
        images, labels = self.read_data()
        # image -> tensor transfrom
        images_pre = [self.Transfrom(img) for img in images]

        # debug
        print('images_pre length', len(images_pre))

        # image data shuffled
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        images_shuffle = []
        labels_shuffle = []

        for index in shuffle_indices:
            print(index)   
            images_shuffle.append(images_pre[index])
            labels_shuffle.append(labels[index])
        
        # 전체 data에 ratio 나누기
        data_size = len(images_shuffle)
        n_train = int(np.ceil(data_size * ratio))
        print('n_train : ', n_train)

        # test
        test_array = np.array(images_shuffle[0:n_train])
        print(test_array)

        return images_shuffle[0:n_train], images_shuffle[n_train + 1:data_size], labels_shuffle[0:n_train], labels_shuffle[n_train + 1:data_size]

    # get anchor, positive, negative images
    def get_triplets(self):
        # index 중 random하게 2개 choice
        label_p, label_n = np.random.choice(self.unique_train_label, 2, replace=False)
        # debug 
        # print('label_p :', label_p)
        # print('label_n :', label_n)

        a, p =  np.random.choice(self.map_train_label_indices[label_p], 2, replace=False)
        n = np.random.choice(self.map_train_label_indices[label_n])

        return a, p, n
    
    # batch size만큼 get triplet images, 보통 batch size 32 64 16 128
    def get_triplets_batch(self, batch_size):
        
        anchor_images, pos_images, neg_images = [], [] ,[]

        for _ in range(batch_size):
            a, p, n = self.get_triplets()
            anchor_images.append(self.images_train[a])
            pos_images.append(self.images_train[p])
            neg_images.append(self.images_train[n])

        # print(self.images_train.shape)
        print('anchor_index : ', len(anchor_images))
        print('pos_index : ', len(pos_images))
        print('neg_index : ', len(neg_images))
        
        return anchor_images, pos_images, neg_images


'''
# proprocess.py file debugging 용 main function
if __name__ == '__main__':

    dataset =  PreProcessing('../data/', 0.9)
    anchor_images, pos_images, neg_images = dataset.get_triplets_batch(16)
    print('anchor_images :' , anchor_images)

'''    




    