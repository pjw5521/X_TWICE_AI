import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
from model_2 import Image_Similarity

class Consine_Similarity():
    
    def __init__(self, model_path):
        self.cuda = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = 0.8
        self.model = Image_Similarity()
        self.model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        self.model.to(self.cuda)
    
    # consine 유사도 구하기
    # 이때 들어오는 vector 값을 numpy array값으로 했다고 가정 
    def forward(self, train_list, noise_list):

        max_list = []
        # similarity 계산
        all_max_info = []

        for vector1 in train_list:
            max_list = []
            max_info = []
            for vector2 in noise_list:
                var_sim = dot(vector1, vector2 ) / (norm(vector1) * norm(vector2))
                max_list.append(var_sim)
            
            # 가장 큰 similarity 값
            max_sim = np.array(max_list)
            max_info.append(max_sim.argmax()) # index
            max_info.append(max_sim.max()) # max value
            
            all_max_info.append(max_info)
        
        return all_max_info
  
        #for idx, vector2 in enumerate(vector2_list):
            #var_sim = dot(vector1, vector2 ) / (norm(vector1) * norm(vector2))
            
            # print("idx: {}, max_sim: {}".format(idx, max_sim))
            # debug
            #print("{} 의 similarity value: {}".format(idx, var_sim))
            
            #sim_list.append(var_sim)
   
    # 딥러닝 모델 Feature vector 반환
    def return_vector(self, img_list):
        
        # print(self.model)
        
        vector_list = []
        
        for img in img_list:
            result = self.model.forward(img)
            result = result.view(-1, 512 * 2* 1 ).cpu()
            result = result.squeeze(0).detach().numpy()
            vector_list.append(result)
        
        return vector_list

