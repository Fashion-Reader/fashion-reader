import os
import cv2
import json
import yaml
import torch
import random
import urllib
import numpy as np
import pandas as pd

""" 공용 함수
    * Logger
    * System
    * Seed
    * Load Crawling Image from DB
"""

"""
Logger
"""


def get_logger(name: str, file_path: str, stream=False) -> logging.RootLogger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


"""
System
"""


def make_directory(directory: str) -> str:
    """경로가 없으면 생성
    Args:
        directory (str): 새로 만들 경로

    Returns:
        str: 상태 메시지
    """

    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            msg = f"Create directory {directory}"

        else:
            msg = f"{directory} already exists"

    except OSError as e:
        msg = f"Fail to create directory {directory} {e}"

    return msg


def count_csv_row(path):
    """
    CSV 열 수 세기
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        n_row = sum(1 for row in reader)


def save_yaml(path, obj):
    try:
        with open(path, 'w') as f:
            yaml.dump(obj, f, sort_keys=False)
        message = f'Json saved {path}'
    except Exception as e:
        message = f'Failed to save : {e}'
    print(message)
    return message


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_json(file_name):
    with open(file_name) as json_file:
        json_data = json.load(json_file)
    return json_data


"""
Seed
"""


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""
Load Crawling Image from DB
"""


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def run_query(query):
    # CONNECT DB
    db = pymysql.connect(host='database.cawpd3yaf0pl.us-east-2.rds.amazonaws.com',
                         user='admin',
                         passwd='00000000')
    cursor = db.cursor()

    # USE DATABASE
    q = """
    use products_table
    """
    cursor.execute(q)

    # CHECK TABLE
    cursor.execute(query)
    out = cursor.fetchall()

    col_names = [i[0] for i in cursor.description]
    df = pd.DataFrame(out, columns=col_names)
    return df


if __name__ == '__main__':
    pass
