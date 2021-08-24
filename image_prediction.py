import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
import torchvision.transforms as transforms


class Image_Prediction():

    def __init__(self, model_path, image):
        self.model = torch.load(model_path)
        self.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        result = self.model.forward(trans_image.unsqueeze(0))
        result = result.view(-1, 512 * 7 * 7)
        result = result.squeeze(0).detach().numpy()

        # result를 numpy 형식으로 반환
        return result, norm(result)

    def Check_Similarity(self, vector_list):

        check = False

        for vector in vector_list:
            var_sim = dot(self.current_vector, vector) / (norm(self.current_vector) * norm(vector))

            # 유사한 경우
            if var_sim > self.threshold:
                check = True

        if check == True:
            return 'Y'
        else :
            return self.current_vector, self.current_norm
