from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus

baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
plusUrl = input('검색어 입력:')
crawl_num = int(input('크롤링할 이미지 개수 입력: '))

# 검색어 url
url = baseUrl + quote_plus(plusUrl)
html = urlopen(url)
soup = bs(html, "html.parser")

img = soup.find_all("img", class_='_img')

print(img)
n = 1

for i in img:
    print(n)
    imgUrl = i['data-source']
    with urlopen(imgUrl) as f:
        with open('./data/train' + str(n) + '.jpg' ,'wb') as h:
            img = f.read()
            h.write(img)
    n += 1
    if(n > crawl_num):
        break

print("Crawling done")
