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
        self.images_train, self.images_mirror, self.images_rotate, self.images_bright, self.images_dark, self.train_labels, self.mirror_labels, self.rotate_labels, self.bright_labels, self.dark_labels = self.preprocessing() 
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
        images_train = []
        images_mirror = []
        images_rotate = []
        images_bright = []
        images_dark = []

        train_labels = []
        mirror_labels = []
        rotate_labels = []
        bright_labels = []
        dark_labels = []
        # dir_name = []

        for dir in os.listdir(self.data_scr):
            # dir_name.append(dir)
            # print(dir)
            try:
                for pic in os.listdir(os.path.join(self.data_scr, dir)):
                    # label create
                    # print('jpg file name; ', pic[:-4])
                    num = re.findall('\d+', pic[:-4])
                    if len(num) == 1:
                        label = int(num[0])
                    elif len(num) == 2:
                        label = int(num[1])

                    if dir == 'train':
                        # image data
                        img = Image.open(self.data_scr + dir + '/' + pic)
                        images_train.append(img)
                        train_labels.append(label)
                    elif dir == 'mirror':
                        img = Image.open(self.data_scr + dir + '/' + pic)
                        images_mirror.append(img)
                        mirror_labels.append(label)

                    elif dir == 'rotate':
                        img = Image.open(self.data_scr + dir + '/' + pic)
                        images_rotate.append(img)
                        rotate_labels.append(label)
                    
                    elif dir == 'dark':
                        img = Image.open(self.data_scr + dir + '/' + pic)
                        images_dark.append(img)
                        dark_labels.append(label)
                    
                    elif dir == 'bright':
                        img = Image.open(self.data_scr + dir + '/' + pic)
                        images_bright.append(img)
                        bright_labels.append(label)
                    
                    
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
        return images_train, images_mirror, images_rotate, images_bright, images_dark, train_labels, mirror_labels, rotate_labels, bright_labels, dark_labels
    
    # data 전처리
    def preprocessing(self):
        
        # 각 images -> PIL type
        images_train, images_mirror, images_rotate, images_bright, images_dark, train_labels, mirror_labels, rotate_labels, bright_labels, dark_labels = self.read_data()
        
        # image -> tensor transfrom
        images_train = [self.Transfrom(img) for img in images_train]
        images_mirror = [self.Transfrom(img) for img in images_mirror]
        images_rotate = [self.Transfrom(img) for img in images_rotate]
        images_bright = [self.Transfrom(img) for img in images_bright]
        images_dark = [self.Transfrom(img) for img in images_dark]

        # debug
        print('images_train length', len(images_train))
        print('images_nosie length', len(images_mirror))
        print('images_nosie length', len(images_rotate))
        print('images_nosie length', len(images_bright))
        print('images_nosie length', len(images_dark))

        '''
        # image data shuffled
        train_shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
        noise_shuffle_indices = np.random.permutation(np.arange(len(noise_labels)))
        train_images_shuffle = []
        noise_images_shuffle = []
        train_labels_shuffle = []
        noise_labels_shuffle = []

        for index in train_shuffle_indices:
            #print(index)   
            train_images_shuffle.append(images_train[index])
            train_labels_shuffle.append(train_labels[index])
        
        for index in noise_shuffle_indices:
            noise_images_shuffle.append(images_noise[index])
            noise_labels_shuffle.append(noise_labels[index])
        '''
        return images_train, images_mirror, images_rotate, images_bright, images_dark, train_labels, mirror_labels, rotate_labels, bright_labels, dark_labels

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
            a, p, n = self.get_triplets() # 각 image의 index값을
            anchor_images.append(self.images_train[a])
            pos_images.append(self.images_train[p])
            neg_images.append(self.images_train[n])

        # print(self.images_train.shape)
        # print('anchor_index : ', len(anchor_images))
        # print('pos_index : ', len(pos_images))
        # print('neg_index : ', len(neg_images))
        
        return anchor_images, pos_images, neg_images

    def val_get_triplets_batch(self, batch_size):

        anchor_images, pos_images, neg_images = [], [], []

        for _ in range(batch_size):
            a, p, n = self.get_triplets
            anchor_images.append(self.images_test[a])
            pos_images.append(self.images_test[p])
            neg_images.append(self.images_test[n])
        
        return anchor_images, pos_images, neg_images



    