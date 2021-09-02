import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch

class Consine_Similarity():
    
    def __init__(self, model_path):
        self.threshold = 0.8
        self.model = torch.load(model_path)
        
    
    # consine 유사도 구하기
    # 이때 들어오는 vector 값을 numpy array값으로 했다고 가정 
    def forward(self, vector1, vector2_list):

        check = False

        # similarity 계산     
        for vector2 in vector2_list:
            var_sim = dot(vector1, vector2 ) / (norm(vector1) * norm(vector2))
            
            if var_sim > self.threshold:
                check = True
                                                                            
        # check 
        if check == True:
            return 'Y' 
        else :
            return 'N'

    def return_vector(self, img_list):
        
        vector_list = []
        
        for img in img_list:
            result = self.model.forward(img)
            print('result: ', result)
            print('result shape: ', result.shape)
            result = result.veiw
            vector_list.append(result)
        
        return vector_list

