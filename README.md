# X_TWICE_AI 개발 명세

## 개발 환경
- 언어 : Python
- 딥러닝 개발 프레임워크 : Pytorch

## 사용 딥러닝 모델
- VGG-16 모델 사용
    - 이미지 분류를 위한 모델
    - 현재 결과: Rotate된 이미지에 대해 정확도가 떨어진다.
- 손실(Loss) 함수 
    - Triplet Loss

## 사용한 라이브러리 버전
- torch==1.8.1
- torchaudio==0.8.1
- torchvision==0.9.1
    - 위의 세개의 라이브러리는 딥 러닝을 위해
- Pillow==8.3.1
    - 이미지의 PIL 파일 형식을 위해
- beautifulsoup4==4.9.3
    - 이미지 웹크롤링을 위해
- numpy==1.19.5
- flask==1.0.3
    - 딥 러닝 서버를 위해 

## 디렉터리 및 파일
- image_create 폴더
    - `crwaling.py` : 이미지 웹크로링을 위한 파일
    - `image_noise_test.py` : 이미지에 noise를 주기 위한 코드 파일
    - `image_noise.py` : 이미지 noise test 코드 파일
- My_model 폴더 : 딥 러닝 pt 파일 저장 폴더
    - `New_Vgg_16.pt` : VGG-16 변형한 딥 모델
- triplet_loss 폴더
    - `train_model.py` : 모델 학습 파일
    - `preprocess.py` : 데이터 전처리 파일
    - `model.py` : 딥러닝 모델 파일
- `image_prediction.py` : BackEnd와 연동하는 python 파일
- `server.py` : Flask를 사용한 AI server 
- `.gitignore` : Git에 올릴 시 무시할 파일 및 파일의 이름이나 경로, 적용 시 `git rm -r --cached .` 명령어 입력
- `data` : 이미지 데이터 파일, gitignore 됨
- `image_similarity_test` : VGG-16 모델을 사용한 결과를 확인
- `model_save.py` : 해당 딥 러닝 모델을 pt 파일로 저장하는 코드 파일 
- `requirements.txt` : 버전 명시 파일

## 이미지 데이터 구축
- `crawling.py` 파일을 통해 구축
- 저작권에 영향이 없도록 무료 이미지 사이트(https://pixabay.com/images/search/)를 통해 이미지 웹 크롤링을 함.
- 이미지 tag가 'src', 'data-lazy-srcset'. 'data-lazy-src'로 되어 있기 때문에 해당 tag에 대해서만 이미지를 가져온다.
- 웹 크롤링을 통해 100개의 이미지를 가져온 후, 다음과 같이 네 개의 변형된 noise를 주었다.
    - train : 원본 데이터
    - rotate : 오른쪽 90도 회전
    - mirror : 좌우반전 이미지
    - bright : 밝기 100
    - dark : 어둡게 50

<br>
<hr>
<br>

# X_TWICE_프로젝트명 서버(Server)

## 서버 환경
- 운영체제 : Ubuntu 18.04 LTS
- GPU : NVIDIA GTX 1080 Ti

## Gunicorn 실행 
1. `pip install gunicorn`으로 gunicorn 설치
2. `gunicorn server:app -b 0.0.0.0 --daemon --access-logfile ./gunicorn-access.log --error-logfile ./gunicorn-error.log`으로 실행. 
    defalt port 번호는 8000으로 모든 아이피에 대해 8000 port 접속 허용. reload 시 같은 명령어 사용 가능.
- `pkill gunicorn` : gunicorn 프로세스 종료
- `./gunicorn-access.log`, `./gunicorn-error.log` 위치에서 access, error log 확인 가능

## torchserve 외부 접속 허용 Port 오픈 시 
1. TorchServe용 구성 파일인 config.properties(기본 이름)를 생성하여 원격 접속 주소 설정
  ```
  inference_address = Inference API 바인딩 주소. Default: http://127.0.0.1:8080
  management_address = Management API 바인딩 주소. Default: http://127.0.0.1:8081
  ```
2. torchserve 실행 시 같은 디렉토리에서 실행하거나 --ts-config으로 경로 지정 
