import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Image_Similarity
import numpy as np
from numpy import dot
from numpy.linalg import norm
# debugging 
from PIL import Image

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cosine_loss = nn.CosineEmbeddingLoss(reduction='none')
transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
img_sim = Image_Similarity().to(cuda)

image1 = Image.open('./cat.jpg')
image1 = transform(image1).to(cuda)

image2 = Image.open('./dog.jpg')
image2 = transform(image2).to(cuda)

result1 = img_sim.forward(image1)
result2 = img_sim.forward(image2) 

#y = torch.ones_like(result1).to(cuda)
y = torch.ones(1).to(cuda)
print("y: ", y)

loss = cosine_loss(result1.view(-1, 1024), result2.view(-1, 1024), y)

print("loss: ", loss)

## calculate cosine sim

vector1 = result1.view(-1, 1024).cpu()
vector1 = vector1.squeeze(0).detach().numpy()


vector2 = result2.view(-1, 1024).cpu()
vector2 = vector2.squeeze(0).detach().numpy()

var_sim = dot(vector1, vector2) / (norm(vector1) * norm(vector2))

print("var_sim: ", 1- var_sim)
