import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class Image_Similarity():

    def __init__(self):
        self.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.old_model = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*(list(self.old_model.children())[0:1])).to(self.cuda)
    
    def forward(self, x):
        return self.model(x.unsqueeze(0))
'''
    # custom loss
    def triplet_loss(self, anchor_img, pos_img, neg_img, margin):
        distance1 = torch.sqrt(torch.sum(torch.pow((anchor_img - pos_img), 2))).to(self.cuda)
        distance2 = torch.sqrt(torch.sum(torch.pow((anchor_img - neg_img), 2))).to(self.cuda)
        # print('distance1 - distance2 : ', distance1 - distance2 + margin) 
        values = max(distance1 - distance2 + margin, torch.Tensor([0]).to(self.cuda)) # torch.max는 return tuple로 두개 , tensor 중 가장 max 값 반환
        print('result_max : ', values)
        print(type(values))
        return values

        
        ## type 변경
        if type(values) == int:
            return torch.FloatTensor(values)
        elif type(values) == torch.Tensor :
            return values
'''



if __name__ == '__main__':
    
    img_sim = Image_Similarity()

    anchor_img = torch.FloatTensor([1,2]).to(img_sim.cuda)
    pos_img = torch.FloatTensor([1,1]).to(img_sim.cuda)
    neg_img = torch.FloatTensor([1,10]).to(img_sim.cuda)

    sim = img_sim.triplet_loss(anchor_img, pos_img, neg_img, 0.5)

    print(sim)
