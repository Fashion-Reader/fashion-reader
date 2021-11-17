import re
import os
import time
import json
import logging
import requests
import datetime


from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from pymongo import MongoClient


# Edit DB Server
client = MongoClient("mongodb+srv://sun:0000@crawling-data.r9oex.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client["crawling-data"]["imvely"]


def download(url, params={}, method='GET', headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}):
    '''
    url crawling
    :param url: 요청받을 url
    :param params: key
    :param headers:
    :param method:
    :param limit:
    :return: response
    '''
    resp = requests.request(method, url,
                            params=params if method == 'GET' else {},
                            data=params if method == 'POST' else {},
                            headers=headers)

    return resp


class ImvelyCrawling():
    def __init__(self, save_dir):
        current_time = datetime.datetime.now().strftime("%Y%m%d %H.%M.%S")
        self.save_dir = os.path.join(save_dir, f'imvely - {str(current_time)}')
        # Edit path
        self.driver = webdriver.Chrome(r'/Users/sun/Imvely/etc/chromedriver')
        self.crawling_items_num = 0
        self.scrapping_items_num = 0
        self.scrapping_items_result = []

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(u'[%(asctime)s] %(levelname)s: %(message)s')

        log_dir = os.path.join(self.save_dir, 'log')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        Log = logging.FileHandler(os.path.join(log_dir, 'imbely.log'), mode='w', encoding='utf8')
        Log.setFormatter(formatter)

        self.logger.addHandler(Log)
        self.logger.info("Start ImvelyCrawling")
        self.logger.info(f"{str(current_time)}")
        
    def save_json(self, save_name):
        save_dir = os.path.join(self.save_dir, save_name)
        with open(save_dir, "w",  encoding="UTF-8") as f:
            json.dump(self.scrapping_items_result, f, indent=4, ensure_ascii=False)
        
    def entered_link(self, link, sleep_time=0.1):
        self.driver.get(link)
        time.sleep(sleep_time)

    def get_main_url(self):
        return 'https://www.imvely.com/'

    def make_cate_urls(self):
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        self.categories = []
        self.total_categories_num = 0
        self.cate_urls = dict([])
        for i in range(6, 11):
            row = soup.select('ul.inline.cut-width > li')[i]
            urls = row.select('a')[1:]
            
            category_list = list(map(lambda x:x.text, urls))
            urls = list(map(lambda x:'https://www.imvely.com' + str(x).split('<a href="')[1].split('">')[0], urls))
            
            for idx, (cate, url) in enumerate(zip(category_list, urls)):
                self.categories.append(cate)
                self.total_categories_num += 1
                self.cate_urls[cate] = url

    def make_cate2item_urls(self):
        self.total_items_num = 0
        self.cate2item_urls = dict([])

        print(self.cate_urls.items())

        for cate, url in self.cate_urls.items():
            st_time = time.time()

            self.entered_link(url)

            html = self.driver.page_source
            soup = BeautifulSoup(html, 'lxml')

            num_of_items = soup.select_one('div.xans-element-.xans-product.xans-product-normalmenu > div > p > strong').text

            items_url = []
            for i in range(int(num_of_items)//40+1):
                self.entered_link(url+f'&page={i+1}')

                html = self.driver.page_source
                soup = BeautifulSoup(html, 'lxml')

                items = soup.select('ul.prdList.grid4 > li.xans-record- > div.thumbnail > a.controllable-link-image')
                items = list(map(lambda x:'https://www.imvely.com' + str(x).split('href="')[1].split('" name=')[0], items))

                items_url.extend(items)

            self.cate2item_urls[cate] = list(set(items_url))
            self.total_items_num += len(list(set(items_url)))

            self.crawling_items_num += 1
            self.logger.info(f"[{self.crawling_items_num}/{self.total_categories_num}] crawling progress time : {(time.time() - st_time):.2f} sec")
            print(f"[{self.crawling_items_num}/{self.total_categories_num}] crawling progress time : {(time.time() - st_time):.2f} sec", end='\r'+' '*100+'\r')
        
    def exceptions(self, x):
        digit_check = re.compile(r'/\d+\.')
        covid_check = re.compile(r'/\d+_19')
        notice_check = re.compile(r'notice')
        modelsize_check = re.compile(r'modelsize')
        fittingsize_check = re.compile(r'size\.')
        blank_check = re.compile(r'/[a-zA-Z]*_\d+\.')
        review_check = re.compile(r'review')
        top_check = re.compile(r'top')
        return '_mj_' in x\
                or digit_check.search(x)\
                or covid_check.search(x)\
                or notice_check.search(x)\
                or modelsize_check.search(x)\
                or fittingsize_check.search(x)\
                or blank_check.search(x)\
                or review_check.search(x)\
                or top_check.search(x)
    
    def remove_selectoption(self, x):
        x = re.sub("\[필수\]|옵션을|선택해|주세요|-| ", "", x)
        return x

    def get_reviews(self, review_pages, item_id):
        print(item_id)
        reviews = None
        for page in range(1, review_pages+1):
            review_url = f"https://sfre-srcs-service.snapfit.co.kr/Dataupdate/GetReview?widgetid=3&platform=pc&Sea=q1MjzaB5IySZ3rdfV0btAUaof2EKEwAJMcKrhHygTvn9UITwMnIFm1ImS5FMZKoe9y3uoM8kaT9N8B0biUxEeEWwc0Ecg84WUnwbilRCLNILrJDPBonWY9jkH0OvxS2%252BjFzeL7tVDHs73VyAzxilsVXrg6%252F8hgTC%252F2Q5iUTB8AdS83uLPbBUH8y8iRpyNATyxxoC5nHtAlvxC%252F5VOyonUBShsbaoHLVLubSgPV0H%252Bls7If2MQKZP6lcYceoMvP8ixsaDMajWbL%252BJtbOLBlyLdA%253D%253D&store_username=XMUv5JHFqw2OqMHINY0dWQ%253D%253D&cateids=%22g3GnyXtaZNM4EI80RGB20gR%252BlSdlxLq85eXS17TDBHTufgK5by32NpoLIhYazrXma4o3B7pBOHI2k0eAKUjv1B%252FTg1SwACR7EUL6OcAhyvEbFupYGrDl0foPigaR21wVlXw31syF71yYqaS3sjHDaq1xCjV4wobuKijS8vhGgPYQaO1UISOsnoYtaiie4b3E3QXEsltBdM32chNSBfz0NJkdQB7hrQiBCuU3MQkCuUltIDoJRixZpcwSGD3V%252B%252Fo9q9fk59qahEAZQeNHo%252FgTW9Ag1448DRgWsXJDWJn0eadkMVuN7lf6DReGy6YRl%252BDQ62aSaNCzUjRxXsP45IDtDw%253D%253D%22&item_id={item_id}&dac=tiaa&iul=0&iub=0&from={page}&filterdata=%257B%2522option%2522%3A%257B%257D%2C%2522userinfofilterdatas%2522%3A%257B%257D%2C%2522scorefilterdatas%2522%3A%255B%255D%2C%2522subscorefilterdatas%2522%3A%257B%257D%2C%2522pricefilter%2522%3A%257B%257D%2C%2522textfilter%2522%3Anull%2C%2522catefilter%2522%3A%2522%2522%2C%2522pivotfilter%2522%3Anull%257D&isrefreshtotalcount=1&order=recent"

            try:
                ret = requests.get(review_url)
            except:
                self.logger.error("requests.exceptions.ConnectionError: ('Connection aborted.', TimeoutError(10060, '연결된 구성원으로부터 응답이 없어 연결하지 못했거나, 호스트로부터 응답이 없어 연결이 끊어졌습니다', None, 10060, None))")
                self.logger.error(f"item_id : {item_id}")
                self.logger.error(f"review_page : {page}")
                self.logger.error(f"review_url : {review_url}")
                continue

            print(ret.content)

            if not ret.content:
                continue

            # ret.content = str(ret.content).strip("'<>() ").replace('\'', '\"')
            parsed = json.loads(ret.content)

            f = lambda x:{'score':x['score'], 'content':x['review_comment'], 'products':x['buy_option']}
            if reviews is None:
                reviews = list(map(f, parsed['data']['reviewinfo']['review']))
            else:
                reviews.extend(list(map(f, parsed['data']['reviewinfo']['review'])))
        return reviews

    def get_item_info(self, url, cate):
        st_time = time.time()
        self.scrapping_items_num += 1
        self.logger.info(f"[{self.scrapping_items_num}/{self.total_items_num}] scrapping progress category : {cate}")
        self.logger.info(f"[{self.scrapping_items_num}/{self.total_items_num}] scrapping progress url : {url}")
        
        self.entered_link(url)

        html = self.driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        try:
            name = soup.select_one('div.infoArea.spec-box > ul > li.name').text
        except:
            return

        try:
            price = soup.select_one('div.prd_price_sale_css.xans-record- > span > span > span').text
        except:
            price = soup.select_one('div.product_price_css.xans-record- > span').text

        option = soup.select('div.infoArea.spec-box > table > tbody.xans-element-.xans-product.xans-product-option.xans-record- > tr > td > select > option')
        option = ' '.join(list(map(lambda x:x.text, option[1:]))).split('empty')

        try:
            colors = option[0].strip().split(' ')
        except:
            colors = []
        colors = list(map(self.remove_selectoption, colors))
        colors = " ".join(list(filter(None, colors)))

        try:
            size = option[1].strip().split(' ')
        except:
            size = []
        size = list(map(self.remove_selectoption, size))
        size = "".join(list(filter(None, size)))

        resp = download(url, method="GET")
        dom = BeautifulSoup(resp.text, 'lxml')
        image = "http:" + dom.find('img', class_ = 'ThumbImage')["src"]

        if image[-3:] != "jpg":
            return

        images = soup.select('div.product-detail.content-box > div > div > img')
        images += soup.select('div.detail_temporary > div > img')
        images += soup.select('div.detail_temporary > div > div > img')
        images += soup.select('div.detail_temporary > div > p > img')
        images += soup.select('div.detail_temporary > div > div > p > img')
        images = list(map(lambda x:'https://www.imvely.com'+str(x).split('src="')[1].split('"/>')[0], images))
        
        images = list(set([image for image in images if not self.exceptions(image)]))

        detail = '\n'.join(map(lambda x:x.text, soup.select('div.new > p')))

        size_info_columns = list(map(lambda x:x.text, soup.select('div.wrap_info.size_info > div.inner > div.tbl > table > thead > tr > th')[1:]))
        size_info_options = list(map(lambda x:x.text, soup.select('div.wrap_info.size_info > div.inner > div.tbl > table > tbody > tr > th')))
        size_info = list(map(lambda x:x.text+'cm' if x.text[-1] != 'g' else x.text, soup.select('div.wrap_info.size_info > div.inner > div.tbl > table > tbody > tr > td')))

        if not size_info_columns:
            size_info_columns += list(map(lambda x:x.text, soup.select('div.size_guide_inner2 > div.size_wrap > div.size_info > table > tbody > tr > td > strong')))
            size_info_options += list(map(lambda x:x.text, soup.select('div.size_guide_inner2 > div.size_wrap > div.size_info > table > tbody > tr > th')))[1:]
            size_info += list(map(lambda x:x.text+'cm' if x.text[-1] != 'g' else x.text, soup.select('div.size_guide_inner2 > div.size_wrap > div.size_info > table > tbody > tr > td')))[len(size_info_columns):]

        size_info_options_num = len(size_info_columns)
        size_info_matrix = []
        for i in range(size_info_options_num):
            size_info_matrix.append([size_info_columns[i]]+size_info[i::size_info_options_num])

        size_infos = None
        for row in zip(['option']+size_info_options, *size_info_matrix):
            if size_infos is None:
                size_infos = {row[0]:list(row[1:])}
            else:
                size_infos[row[0]] = list(row[1:])


        washing_and_care = list(map(lambda x:x.text, soup.select('div.wrap_info.care_info > div.inner > ul.list > li > p')))
        washing_and_care += list(map(lambda x:x.text, soup.select('div.wrap_info.care_info > div.etc > ul > li')))
        if not washing_and_care:
            washing_and_care = list(map(lambda x:x.text, soup.select('div.washing_tip_wrap > div.inner > ul.list > li > p')))
            washing_and_care += list(map(lambda x:x.text, soup.select('div.washing_tip_wrap > div.etc > ul > li')))
        
        fabric = {"두계감" :"", "비침": "", "신축성": "", "촉감": "", "핏": "", "안감": ""}, 
        fabric = {t.text:f.text for t, f in zip(soup.select('div.table_inner.fabric_inner > ul > li.title'),
                                                soup.select('div.table_inner.fabric_inner > ul > li.on'))}
        if not fabric:
            fabric = {t.text:f.text for t, f in zip(soup.select('div.wrap_info.fabric_info > div.inner > div.tbl > table > tbody > tr > th'),
                                                    [x for x in soup.select('div.wrap_info.fabric_info > div.inner > div.tbl > table > tbody > tr > td') if x.select('span.check')])}

        product_info = list(map(lambda x:x.text, soup.select('div.table_inner.pro_info_inner > ul > li')))
        product_info = {t:p for t, p in zip(product_info[::2], product_info[1::2])}

        if not product_info:
            product_info = {t.text:p.text for t, p in zip(soup.select('div.wrap_info.product_info > div.inner > div.tbl > table > tbody > tr > th'),
                                                soup.select('div.wrap_info.product_info > div.inner > div.tbl > table > tbody > tr > td'))}

        review_nums = int(''.join(soup.select_one('div.infoArea.spec-box > ul > li > a > span.b_count.snap_review_count.noset').text.split(',')))
        
        # review_pages = math.ceil(review_nums/10)
        item_id = re.findall(r'product_no=(\d{1,5})', url)[0]
        # reviews = self.get_reviews(review_pages, item_id)
        # reviews = None

        try:
            item_info = {'item_id':item_id,
                        'name':name,
                        'url':url,
                        'item_type':cate,
                        'item_type_id': url.split("=")[2].split("&")[0],
                        'price':price,
                        'size_options':size,
                        'color_options':colors,
                        'item_img_links':image,
                        'detail':detail,
                        'size_infos':size_infos, 
                        'thickness': fabric["두께감"],
                        'see_through': fabric["비침"],
                        'flexibility': fabric["신축성"],
                        'touch': fabric["촉감"],
                        'fit': fabric["핏"],
                        'lining': fabric["안감"]}
                        # 'review_nums':review_nums,
                        # 'review':reviews}
        except:
            return
        self.scrapping_items_result.append(item_info)

        self.logger.info(f"[{self.scrapping_items_num}/{self.total_items_num}] scrapping progress time : {(time.time() - st_time):.2f} sec")
        print(f"[{self.scrapping_items_num}/{self.total_items_num}] scrapping progress time : {(time.time() - st_time):.2f} sec", end='\r'+' '*100+'\r')

        if self.scrapping_items_num % 10 == 0:
            db.insert_many(self.scrapping_items_result)
            save_name = f"scrapped_result_{self.scrapping_items_num}.json"
            # self.save_json(save_name)
            self.scrapping_items_result = []

            self.logger.info(f"Save {save_name}")
            print(f"\tSave {save_name}")
    
    def scrapping(self):
        for cate in self.categories:
            print(cate)
            for item_url in tqdm(self.cate2item_urls[cate]):
                self.get_item_info(item_url, cate)

        if self.scrapping_items_result:
            save_name = f"scrapped_result_{self.scrapping_items_num}.json"
            # self.save_json(save_name)
            self.scrapping_items_result = []
            
            self.logger.info(f"Save {save_name}")
            print(f"\tSave {save_name}")


def main(save_dir):
    imvely = ImvelyCrawling(save_dir)
    url = imvely.get_main_url()
    imvely.entered_link(url, 3)

    imvely.logger.info('Start Crawling !!')
    print('Start Crawling !!')

    imvely.make_cate_urls()
    imvely.make_cate2item_urls()

    imvely.logger.info("Success Crawling !!")
    print("Success Crawling !!")

    imvely.logger.info('Start Scrapping !!')
    print('Start Scrapping !!')

    imvely.scrapping()

    imvely.logger.info('Success Scrapping !!')
    print('Success Scrapping !!')


if __name__ == "__main__":
    save_dir = './'
    main(save_dir)
