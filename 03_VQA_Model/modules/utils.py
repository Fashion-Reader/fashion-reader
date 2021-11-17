""" 공용 함수

"""

import os
import cv2
import json
import yaml
import torch
import random
import urllib
import numpy as np


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_json(file_name):
    with open(file_name) as json_file:
        json_data = json.load(json_file)
    return json_data


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
