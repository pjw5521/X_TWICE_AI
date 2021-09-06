import torch
from torch import cuda
import torch.optim  as optim
import torch.nn  as nn
from preprocess import PreProcessing
from model_2 import Image_Similarity
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle


if __name__ == "__main__":

    # torch cache 삭제
    torch.cuda.empty_cache()
    
    # 변수 설정
    data_scr = '../data/'
    ratio = 0.9
    batch_size = 16
    train_iter = 141
    step = 50 # batch size당 몇번 학습 사용 x
    lr = 0.0001 
    margin = 0.5
    epoch_list = []
    val_epoch_list = []
    train_loss_list = []
    val_loss_list = []

    # data & model & Gpu Setup
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Dataset = PreProcessing(data_scr, ratio)
    image_sim = Image_Similarity().to(cuda)
    next_batch = Dataset.get_triplets_batch
    test_images = Dataset.images_test # validation image data
    test_labels = Dataset.labels_test # validation image data

    # model setting 
    cosine_loss = nn.CosineEmbeddingLoss(reduction='none')
    optimizer = optim.SGD(image_sim.parameters(), lr = lr , momentum= 0.99)

    ## 초기 parameters 값
    print('initail model parameter: {}'.format(image_sim.parameters()))

    # model train
    for epoch in range(train_iter):
        # debug
        print('epoch : ', epoch)
        epoch_list.append(epoch)
        
        # model train mode
        image_sim.train()

        anchor_images, pos_images, neg_images = next_batch(batch_size)
        val_anchor_images, val_pos_images, val_neg_images = next_batch(batch_size)
        batch_train_loss = 0
        batch_val_loss = 0

        # print('anchor_image : ', anchor_images[0])
        for anchor_img, pos_img, neg_img in zip(anchor_images, pos_images, neg_images):

            optimizer.zero_grad() # gradient to zero
            # input network
            anchor_output = image_sim.forward(anchor_img) # 512 * 2 * 1
            pos_output = image_sim.forward(pos_img)
            neg_output = image_sim.forward(neg_img)
            y = torch.ones_like(anchor_output.view(-1, 512* 2 * 1))
            pos_loss = cosine_loss(anchor_output.view(-1, 512* 2 * 1), pos_output.view(-1, 512* 2* 1),  y)
            neg_loss = cosine_loss(anchor_output.view(-1, 512* 2 * 1), neg_output.view(-1, 512* 2* 1), -1 * y)
            # debug
            print("pos_loss: ", pos_loss)
            print("neg_loss: ", neg_loss)

            loss = cosine_loss(anchor_output.view(-1, 512* 2 * 1), pos_output.view(-1, 512* 2* 1),  y) + cosine_loss(anchor_output.view(-1, 512* 2 * 1), neg_output.view(-1, 512* 2* 1), -1 * y)
            batch_train_loss += loss
            # print('epoch: {} , loss: {}'.format(epoch, loss))

            if not torch.isfinite(loss):
                print('non - finite loss : loss가 유한값이 아님')
                exit(1)

            loss.backward() # backpropagation
            optimizer.step() # update gradient

        # error save
        train_loss_list.append((batch_train_loss /  batch_size).detach().cpu().numpy())
        
        # validation
        if epoch % 20 == 0:
            val_epoch_list.append(epoch)
            image_sim.eval()
            with torch.no_grad():
                for anchor_img, pos_img, neg_img in zip(val_anchor_images, val_pos_images, val_neg_images):

                    # input network
                    val_anchor_output = image_sim.forward(anchor_img)
                    val_pos_output = image_sim.forward(pos_img)
                    val_neg_output = image_sim.forward(neg_img)

                    v_loss = cosine_loss(val_anchor_output.view(-1, 5112* 2 * 1), val_pos_output.view(-1, 512* 2* 1),  y) + cosine_loss(val_anchor_output.view(-1, 5112* 2 * 1), val_neg_output.view(-1, 512* 2* 1), -1 * y)
                    #v_loss = cosine_loss(val_anchor_output, val_pos_output, val_neg_output)
                    batch_val_loss += v_loss
                
                val_loss_list.append((batch_val_loss / batch_size).detach().cpu().numpy())
                # error 출력
                print("epoch: {}/ {} | train_loss: {} | val_loss: {}".format(epoch, train_iter, batch_train_loss / batch_size, batch_val_loss / batch_size))
        else: 
            continue

    print("successfully train end")
    
    ## result 
    print('train_loss: ', train_loss_list)
    print('val_loss: ', val_loss_list)
    print('epoch_lis :', epoch_list)
    print('val_epoch_list', val_epoch_list)

    
    # error list save
    with open('./train_loss_list_3.txt', 'wb') as f:
        pickle.dump(train_loss_list, f)
    
    with open('./val_loss_list_3.txt', 'wb') as f:
        pickle.dump(val_loss_list, f)
    

    fig = plt.figure(figsize=(12,5))
    # train_loss
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('loss 추이')
    ax1.plot(epoch_list, train_loss_list)
    plt.legend(['train loss'])

    # validation_loss
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('loss 추이')
    ax2.plot(val_epoch_list, val_loss_list)
    plt.legend(['val loss'])

    # validation vector scatter
    ax3 = fig.add_subplot(1, 3, 3)
    val_results = []
    val_labels = []

    with torch.no_grad():
        for img, label in zip(test_images, test_labels):
            # print('image_result: ', image_sim.forward(img))
            val_results.append(image_sim.forward(img).detach().cpu().numpy())
            val_labels.append(label)

    # val_results = np.concatenate(val_results)
    val_results = np.array(val_results)
    val_labels = np.array(test_labels)

    print('val_result_shape: ', val_results.shape)
    # print('val_result_0 : ', val_results[0])
    print('val_labels : ', val_labels)

    for label in np.unique(val_labels):
        tmp = val_results[val_labels == label]
        #print('tmp type: ', type(tmp))
        plt.scatter(tmp[:, 0], tmp[:, 1], label = label)
    
    plt.legend()
    plt.show()

    ## 초기 parameters 값
    print('After train model parameter: {}'.format(image_sim.parameters()))
    

    ## model_save
    print('model_save')
    # torch.save(image_sim.state_dict() , '../My_model/cosine_Vgg_1.pt') 
    # (lr : 0.0001, iter: 200 -> 512_1), (r : 0.0001, iter:141... -> 512_2), (r : 0.0001, iter: 61 -> 512_3), lr을 0.0001보다 크게 할 때, loss: infinite
    # (lr : 0.0002,iter: 141 -> 512_4), (lr : 0.0001,iter: 141 -> 512_4, batch-> 32 -> 512_5)
    # (r : 0.0001, iter:141... -> 512_6)

    ## Max
    # (lr : 0.0001, iter:141... -> Max_Vgg_512_1), (lr : 0.0001, iter:201... -> Max_Vgg_512_2), (lr : 0.0002,iter: 141 -> 512_3)
    # (lr : 0.0001,iter: 141 -> 512_4, batch-> 32 -> 512_4)
    print("model save successfully")

             

        


