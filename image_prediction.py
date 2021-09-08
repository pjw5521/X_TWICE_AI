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
        self.threshold = 0.98
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
        token_id = '1'
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
        count = 0 
        #차원 맞추기 
        for vector in convert_vector_list:
            var_sim = dot(self.current_vector, vector) / (norm(self.current_vector) * norm(vector))
            # 유사한 경우 
            if var_sim > self.threshold:
                test = vector_list[count]
                token_id = db_connection.getTokenId(test)
                check = True
                
            count = count + 1; 

        if check == True:
            print("check : yes")
            result = 'Y'
            #os.remove('image.jpg')
            return result, token_id

        else :
            print('check : No')
            #os.remove('image.jpg')
            #return dense_vector.data, self.current_norm
            return self.current_vector, self.current_norm,

'''
if __name__ == '__main__':

    image  = Image.open('./test/train/train5.jpg')
    image2 = Image.open('./test/mirror/mirror5.jpg')
    image3 = Image.open('./test/train/train27.jpg')

    predict = Image_Prediction('./My_model/cosine_Vgg_3.pt', image)
    print('predict current_vector :', predict.current_vector)
    print('norm', predict.current_norm )
    result = predict.Check_Similarity()
    print("result: ", result[0] == 'Y')

    predict2 = Image_Prediction('./My_model/cosine_Vgg_3.pt', image2)
    print('predict current_vector :', predict2.current_vector)
    print('norm', predict2.current_norm )
    print("result: ", predict2.Check_Similarity())

    predict3 = Image_Prediction('./My_model/cosine_Vgg_3.pt', image3)
    print('predict current_vector :', predict3.current_vector)
    print('norm', predict3.current_norm )
    print("result: ", predict3.Check_Similarity())

    var_sim = dot(predict.current_vector, predict3.current_vector) / (norm(predict.current_vector) * norm(predict3.current_vector))
    print("var_sim: ", var_sim)

'''