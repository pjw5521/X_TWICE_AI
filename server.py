import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import gunicorn
from image_prediction import Image_Prediction
from PIL import Image
import numpy

app = Flask(__name__)
imagenet_class_index = json.load(open('./imagenet_class_index.json'))

PATH = './My_model/New_Vgg_16.pt'
#model = torch.load(PATH) 
# PATH 수정 
model = models.densenet121(pretrained=True)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    # print(request)
    if request.method == 'POST':
        #print(request.files)
        file = request.files['file']
        #img_bytes = file.read()
        img = Image.open(file)
        #image  = Image.open('./image.png')
        #class_id, class_name = get_prediction(image_bytes=img_bytes)
        #return jsonify({'class_id': class_id, 'class_name': class_name})
        #prediction = Image_Prediction(PATH, image_bytes=img_bytes)
        prediction = Image_Prediction(PATH, img)
        result = prediction.Check_Similarity()
       
        if result == 'Y': 
            return jsonify({ 'result' : 'fail' })
        else :
            with numpy.printoptions(threshold=numpy.inf):
                vertorlist = str(result[0])
            return jsonify({ 'picture_vector': vertorlist, 'picture_norm' : str(result[1]) })
        

if __name__ == '__main__':
    # app.run()
    #app.run(host='0.0.0.0')

    image  = Image.open('./image.png')
    predict = Image_Prediction('./My_model/New_Vgg_16.pt', image)
    #with numpy.printoptions(threshold=numpy.inf):
    #    print('predict current_vector :', predict.current_vector)
    print('vertor_norm : ', predict.current_norm )
    result = predict.Check_Similarity()
    with numpy.printoptions(threshold=numpy.inf):
        print("result : ", result[0])

   
######################################################################
# 이제 웹 서버를 테스트해보겠습니다! 다음과 같이 실행해보세요:
# FLASK_ENV=development python3 server.py flask run
# gunicorn server:app -b 0.0.0.0 --daemon

#######################################################################
# `requests <https://pypi.org/project/requests/>`_ 라이브러리를 사용하여
# POST 요청을 만들어보겠습니다:
#
# .. code-block:: python
#
#import requests
#
# resp = requests.post("http://localhost:5000/predict",
#                     files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})

#######################################################################
# `resp.json()` 을 호출하면 다음과 같은 결과를 출력합니다:
#
# ::
#
#  {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#
