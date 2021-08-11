import logging # 어떤 소프트웨어가 실행될 때 발생하는 이벤트를 추적하는 수단
import torch
import torch.nn.functional as F
import io
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
import torchvision.transforms as transforms
import numpy

class MyHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # 하나의 이미지 전처리
    def preprocess_image(self, req):

        image = req.get("data")
        if image is None:
            image = req.get("body")
        
        # PIL Image open
        image = Image.open(io.BytesIo(image))
        image = self.transform(image)

        return image.unsqueeze(0) 
    
    # preprocess: 여러개의 요청 처리
    def preprocess(self, requests):

        images = [self.preprocess_image(req) for req in requests]
        images = torch.cat(images)

        return images

    # model의 result 값 retur: torch.tensor
    def inference(self, image):

        result = self.model(image)
        return result.view(-1, 512 * 7 * 7)

    # 후처리: list형식으로 return
    # numpy 형식의 vector로 
    def postprocess(self, preds):

        res = []

        for pred in preds:
            pred = pred.cpu()
            res.append({'vector': pred.squeeze(0).detach().numpy})
        
        return res





        


    






    

