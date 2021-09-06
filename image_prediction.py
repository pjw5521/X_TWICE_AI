import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
import torchvision.transforms as transforms
from PIL import Image
import db_connection
from model import Image_Similarity
import os 

class Image_Prediction():

    def __init__(self, model_path, image):

        self.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Image_Similarity()
        self.model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        self.model.to(self.cuda)
        self.transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.threshold = 0.8
        self.current_vector, self.current_norm = self.GetFeatureVector(image)

    # image가 PIL image일 일때
    def GetFeatureVector(self, image):
        
        trans_image = self.transform(image).to(self.cuda)
        result = self.model(trans_image)
        result = result.view(-1, 1024).cpu()
        result = result.squeeze(0).detach().numpy()

        # result를 numpy 형식으로 반환
        return result, norm(result)

    def Check_Similarity(self):

        check = False
        vector_list = db_connection.select_vector(self.current_norm)
        
        convert_vector_list = []
        # string to array 
        for i in range(len(vector_list)):
            convert_vector_list.append(vector_list[i].split(', '))
        
        # string array to float array 
        convert_vector_list= np.array(convert_vector_list, dtype=float)
        #print(convert_vector_list)
        print("current.vector shape", self.current_vector.shape)
        ''' vector 압축
        dense_vector = csr_matrix(self.current_vector).reshape(1,-1)
        print("dense_vector shape : ", dense_vector.shape )
        '''
        return self.current_vector, self.current_norm
        '''
        #차원 맞추기 
        for vector in convert_vector_list:
            var_sim = dot(self.current_vector, vector) / (norm(self.current_vector) * norm(vector))
            print("var sim ", var_sim)
            # 유사한 경우 
            if var_sim > self.threshold:
                check = True

        if check == True:
            print("check : yes")
            os.remove('image.jpg')
            return 'Y'

        else :
            print('check : No')
            os.remove('image.jpg')
            #return dense_vector.data, self.current_norm
            return self.current_vector, self.current_norm
        '''
'''
if __name__ == '__main__':

    image  = Image.open('./image.png')
    predict = Image_Prediction('./My_model/New_Vgg_16.pt', image)
    print('predict current_vector :', predict.current_vector)
    print('norm', predict.current_norm )
'''    