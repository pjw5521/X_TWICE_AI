import torch
import torch.nn  as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request
from PIL import Image

def url_to_image(url):
  resp = urllib.request.urlretrieve(url, "image.png")
  #image = np.asarray(bytearray(resp.read()), dtype="uint8")
  #image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  image = Image.open("image.png")

  return image

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

model = models.vgg16(pretrained=True)
New_model = nn.Sequential(*(list(model.children())[0:1]))
#New_model = nn.Sequential(*(list(model.children())[0:1]),
#                          torch.squeeze(1),
#                          nn.AdaptiveAvgPool1d(output_size=1000)     
#                          ) # 512 x 7 x 7
print(New_model)
New_model.eval()
# example input

img_url = "https://www.sjpost.co.kr/news/photo/202007/53199_48342_4214.jpg"
img = url_to_image(img_url)
img = transform(img) # example input

result = New_model(img.unsqueeze(0))
print(result)
print(result.shape)
# model save
# torch.save(New_model, './My_model/New_Vgg_16.pt',  _use_new_zipfile_serialization=False )
# traced_model = torch.jit.trace(New_model, img.unsqueeze(0))
# traced_model.save("./New_Vgg_16.pt")

