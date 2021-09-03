import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
# debugging 
from PIL import Image
import torchvision.transforms as transforms

class Image_Similarity(nn.Module):

    def __init__(self):
        super(Image_Similarity, self).__init__()
        self.layer1 = nn.Sequential(*(list(models.vgg16(pretrained=True).children())[0:1]))
        self.layer2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):

        result = self.layer1(x.unsqueeze(0))
        result = self.layer2(torch.squeeze(result))
        return  result

    
