from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.request import Request
import urllib.request
from urllib.parse import quote_plus
import os
import sys
from selenium import webdriver
from time import sleep

# 검색어 및 이미지 개수 설정
keyword = input('검색어 입력:')
crawl_num = int(input('크롤링할 이미지 개수 입력: '))

pages = int((crawl_num-1)/100)+1 # pages 개수
img_count = 1 # image count
finish = False

# image 저장 경로
path = './data/train'

# 크롬 드라이버 설정
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('lang=ko_KR')
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36")
driver = webdriver.Chrome('../chromedriver.exe',chrome_options=chrome_options)

for i in range(1, int(pages)+1):
    # 구글 웹페이지에 로드
    driver.get('https://pixabay.com/images/search/'+keyword+'/?pagi='+str(i))
    sleep(1)
    
    html = driver.page_source
    soup = bs(html, "html.parser")

    imgs = soup.select('div.flex_grid.credits.search_results img') #요소 선택

    # 마지막 이미지 여부
    lastPage = False
    if len(imgs) != 100:
        lastPage = True

    for img in imgs:
        # 이미지 개수 증가
        img_count += 1

        srcset = ""    
        # get 한 image에 srcset이라는 속성이 없다면 -> data-lazy-srcset
        if img.get('srcset') == None:
            srcset = img.get('data-lazy-srcset')
        else:
            srcset = img.get('srcset')

        if len(srcset):

            src = str(srcset).split()[0] # 가장 작은 이미지 경로 추출
            print(str(i) + '번째 이미지 경로')
            print(src)

            # user-agent 헤더 request
            req = Request(src, headers={'User-Agent': 'Mozilla/5.0'})

            try:
                imgUrl = urlopen(req).read()
                with open(path + str(img_count)+'.jpg','wb') as f:
                    f.write(imgUrl) # 파일 저장
            except urllib.error.HTTPError:
                print('error')
                sys.exit(0)
        
        # 입력된 이미지 개수만큼 크롤링 했을 경우
        if img_count == crawl_num:
            finish = True
            break
    
    if finish or lastPage:
        break

print('크롤링된 이미지 cont' + str(img_count))





