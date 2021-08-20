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
- `crwaling.py` : 이미지 웹크로링을 위한 파일
- `image_noise_test.py` : 이미지에 noise를 주기 위한 코드 파일
- `image_noise.py` : 이미지 noise test 코드 파일
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

## NodeJS 설치
1. `curl -sL https://deb.nodesource.com/setup_10.x | sudo bash -` 으로 NodeJS 10 저장소 위치를 변경
2. `sudo apt install nodejs` 으로 node js 설치
3. `node -v`로 NodeJS 버전 확인하고, `npm -v`으로 NPM 버전 확인
4. `sudo npm install -g yarn pm2`으로 전역으로 Yarn과 PM2를 설치


