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


# debug
if __name__ == '__main__':

    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    image = Image.open('../../cat.jpeg')
    image = transform(image).to(cuda)
    print(image)

    img_sim = Image_Similarity().to(cuda)
    result = img_sim.forward(image)
    result = result.view(-1, 512 * 1 * 1)
    print('result: ', result)
    print('result shape: ', result.squeeze().shape)

    # model save
    # torch.save(img_sim, '../My_model/New_Vgg_512.pt')

