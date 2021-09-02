import torch
import numpy as np
from preprocess_test import PreProcessing
from cosine_similarity import Consine_Similarity


if __name__ == '__main__':

    # 변수 설정
    data_src = '../test/'
    ratio = 1.0 # 전체 test image를 가져와야함
    model_path_1 = '../My_model/train_Vgg_512_1.pt'

    # data setting
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Dataset = PreProcessing(data_src, ratio)

    images_train = Dataset.images_train
    images_noise = Dataset.images_noise
    labels_train = Dataset.labels_train
    labels_noise = Dataset.labels_noise

    # debug
    print('train_images: ', len(images_train))
    print('train_labels: ', len(labels_train))
    print('noise_images: ', len(images_noise))
    print('labels_noise: ', len(labels_noise))

    # model load
    cosine_sim = Consine_Similarity(model_path_1)

    train_vector = cosine_sim.return_vector(images_train)
    noise_vector = cosine_sim.return_vector(images_noise)
    print(len(train_vector))
    print('noise_vector: ', len(noise_vector))

    ## circulate cosine_sim

    image1_sim, image1_label = cosine_sim.forward(train_vector[0], noise_vector)
    print('image1_sim: ', image1_sim)
    print('image1_label: ',image1_label)






