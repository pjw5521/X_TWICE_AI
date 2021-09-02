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

        max_sim = []
        max_label = []
        
        # similarity 계산     
        for idx, vector2 in enumerate(vector2_list):

            var_sim = dot(vector1, vector2 ) / (norm(vector1) * norm(vector2))

            if len(max_sim) < 4:
                max_sim.append(var_sim)
                max_label.append(idx)
            else:
                for i in range(4):
                    if var_sim > max_sim[i]:
                        max_sim[i] = var_sim
                        max_label[i] = idx

        return max_sim, max_label
            

    def return_vector(self, img_list):
        
        vector_list = []
        
        for img in img_list:
            result = self.model.forward(img)
            result = result.view(-1, 512).cpu()
            result = result.squeeze(0).detach().numpy()
            vector_list.append(result)
        
        return vector_list

