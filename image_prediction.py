import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch
import torchvision.transforms as transforms
from PIL import Image
import db_connection
from scipy.sparse import csr_matrix 

class Image_Prediction():

    def __init__(self, model_path, image):

        self.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path).to(self.cuda)
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
        result = self.model(trans_image.unsqueeze(0))
        result = result.view(-1, 512 * 7 * 7).cpu()
        result = result.squeeze(0).detach().numpy()

        # result를 numpy 형식으로 반환
        return result, norm(result)

    def Check_Similarity(self):

        check = False
        vector_list = db_connection.select_vector(self.current_norm)
          
        # string list to float list
        vector_list= np.array(vector_list,dtype=float) 
        #print(vector_list)

        dense_vector = csr_matrix(self.current_vector)
        #densevector = densevector.toarray() 

        with np.printoptions(threshold=np.inf):
            print(dense_vector.data)

        print("densevector.data.shape : ", dense_vector.data.shape)
        
        for vector in vector_list:
            var_sim = dot(self.current_vector, vector) / (norm(self.current_vector) * norm(vector))

            # 유사한 경우 
            if var_sim.any() > self.threshold:
                check = True

        if check == True:
            print("check : yes")
            return 'Y'
        else :
            print('check : No')
            return dense_vector, self.current_norm

'''
if __name__ == '__main__':

    image  = Image.open('./image.png')
    predict = Image_Prediction('./My_model/New_Vgg_16.pt', image)
    print('predict current_vector :', predict.current_vector)
    print('norm', predict.current_norm )
'''    