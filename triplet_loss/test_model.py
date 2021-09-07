import torch
import numpy as np
from preprocess_test import PreProcessing
from cosine_similarity import Consine_Similarity

class Model_accuracy():

    def __init__(self):

        pass

    def test_accuracy(self):
        return None


##########################################################################

if __name__ == '__main__':

    # 변수 설정
    data_src = '../test/'
    ratio = 1.0 # 전체 test image를 가져와야함

    model_path_1 = '../My_model/Max_Vgg_512_1.pt'
    model_path_2 = '../My_model/Max_Vgg_512_2.pt'
    model_path_3 = '../My_model/Max_Vgg_512_3.pt'
    model_path_4 = '../My_model/Max_Vgg_512_4.pt'
    model_path = '../My_model/cosine_Vgg_3.pt'

    # data setting -> no shufflge
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Dataset = PreProcessing(data_src, ratio)

    images_train = Dataset.images_train
    images_mirror = Dataset.images_mirror
    images_rotate = Dataset.images_rotate
    images_bright = Dataset.images_bright
    images_dark = Dataset.images_dark

    labels_train = Dataset.train_labels
    labels_mirror = Dataset.mirror_labels
    labels_rotate = Dataset.rotate_labels
    labels_bright = Dataset.bright_labels
    labels_dark = Dataset.dark_labels 

    # debug
    print('train_images: ', len(images_train))
    print('train_labels: ', labels_train)

    print('mirror_images: ', len(images_mirror))
    print('mirror_labels: ', labels_mirror)

    print('rotate_images: ', len(images_rotate))
    print('rotate_labels: ', len(labels_rotate))

    print('bright_images: ', len(images_bright))
    print('bright_labels: ', len(labels_bright))

    print('dark_images: ', len(images_dark))
    print('dark_labels: ', len(labels_dark))

    # model load
    cosine_sim = Consine_Similarity(model_path)
    #cosine_sim_1 = Consine_Similarity(model_path_1)
    #cosine_sim_2 = Consine_Similarity(model_path_2)
    #cosine_sim_3 = Consine_Similarity(model_path_3)
    #cosine_sim_4 = Consine_Similarity(model_path_4)

    train_vector_1 = cosine_sim.return_vector(images_train)
    mirror_vector_1 = cosine_sim.return_vector(images_mirror)
    rotate_vector_1 = cosine_sim.return_vector(images_rotate)
    bright_vector_1 = cosine_sim.return_vector(images_bright)
    dark_vector_1 = cosine_sim.return_vector(images_dark)

    print('train vector: ', train_vector_1[0])
    print("train_vector avg:", np.average(train_vector_1[0]))

    ## circulate cosine_sim
    image1_sim = cosine_sim.forward(train_vector_1, bright_vector_1) 
    print(image1_sim)
    #image1_sim = np.array(image1_sim)

    '''
    for i in range(4):
        idx = image1_sim.argmax()
        max = image1_sim.max()
        image1_sim[idx] = 0
        # print("noise_label: ", labels_noise[idx])
        print("idx: {}, max: {} ".format(idx, max))  

    # debug
    print("image1_sim: ", image1_sim)  
    '''
    





