import logging
import io
import os
import torch
from ts.torch_handler.base_handler import BaseHandler
import torchvision.transforms as transforms
from PIL import Image


class ModelHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # initailize: model load & device 설정
    def initialize(self, context):

        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # device 설정
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu" )

        # Read pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        
        # load the model
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        '''
        if self.model is None:
            logging.Logger.debug("loaded fail")
        else:
            logging.Logger.debug("미네: model file %s loaded successfully", model_pt_path) 
        
        '''
        
        self.model.eval()
        self.initialized = True

    
    # 하나의 req_image 처리
    def preprocess_image(self, req):
        
        image = req.get("data")
        if image is None:
            image = req.get("body")
        
        print(type(image))
        # PIL Image open
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)

        return image.unsqueeze(0).to(self.device) 
    
    # preprocess: 여러개의 요청 처리
    def preprocess(self, requests):

        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        images = [self.preprocess_image(req) for req in requests]
        images = torch.cat(images)

        return images
    
    def inference(self, model_input):

        """
        Internal inference methods
        :param model_input: transformed model input data 하나의 input?
        :return: list of inference output in NDArray
        + image feature tensor vector가 return
        """
        model_output = self.model.forward(model_input)
        return model_output.view(-1, 512 * 7 * 7)
    
    def postprocess(self, preds):

        res = []

        for pred in preds:
            pred = pred.cpu()
            res.append({'vector': pred.squeeze(0).detach().numpy})

        return res
    
    def handle(self, data, context):

        model_input = self.preprocess(data)
        model_output = self.inference(model_input)

        return self.postprocess(model_output)
        


    





