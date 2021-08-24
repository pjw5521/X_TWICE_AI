import torch
import torch.optim  as optim
from preprocess import PreProcessing
from model import Image_Similarity
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == "__main__":

    # 변수 설정
    data_scr = '../data/'
    ratio = 0.9
    batch_size = 16
    train_iter = 1
    step = 50 # batch size당 몇번 학습
    lr = 0.01
    margin = 0.5

    # data setUp & model Setup
    Dataset = PreProcessing(data_scr, ratio)
    image_sim = Image_Similarity()
    next_batch = Dataset.get_triplets_batch

    # model setting 
    # loss = model.triplet_loss 
    optimizer = optim.SGD(image_sim.model.parameters(), lr = lr , momentum= 0.5)

    # model train
    for epoch in range(train_iter):
        
        image_sim.model.train()

        anchor_images, pos_images, neg_images = next_batch(batch_size)

        for anchor_img, pos_img, neg_img in zip(anchor_images, pos_images, neg_images):
            # debug
            print('anchor len : ', len(anchor_img))

            optimizer.zero_grad() # gradient to zero

            anchor_output = image_sim.forward(anchor_img)
            pos_output = image_sim.forward(pos_img)
            neg_output = image_sim.forward(neg_img)

            loss = image_sim.triplet_loss(anchor_output, pos_output, neg_output, margin)
            loss.backward() # backpropagation

            optimizer.step()

        


