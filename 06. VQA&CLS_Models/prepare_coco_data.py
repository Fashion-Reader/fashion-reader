from collections import OrderedDict
import os
import json
import funcy
from sklearn.model_selection import train_test_split
from modules.utils import load_json

label_paths = glob.glob('./라벨링데이터/*/*')

file_data = OrderedDict()

file_data['info'] = {}
file_data['licenses'] = [{
    "id": 0,
    "name": "won",
            "url": ""
}]

lst = []

cat_list = {'티셔츠': 0, '탑': 1, '후드티': 2, '니트웨어': 3, '셔츠': 4, '블라우스': 5,
            '브라탑': 6, '팬츠': 7, '조거팬츠': 8, '청바지': 9, '래깅스': 10, '스커트': 11, '패딩': 12, '짚업': 13,
            '점퍼': 14, '재킷': 15, '베스트': 16, '가디건': 17, '코트': 18, '드레스': 19, '점프수트': 20, 'UNKNOWN': 21}

cat = [{
    "id": 0,
    "name": "티셔츠",
            "supercategory": "상의"
}, {
    "id": 1,
    "name": "탑",
            "supercategory": "상의"
}, {
    "id": 2,
    "name": "후드티",
            "supercategory": "상의"
}, {
    "id": 3,
    "name": "니트웨어",
            "supercategory": "상의"
}, {
    "id": 4,
    "name": "셔츠",
            "supercategory": "상의"
}, {
    "id": 5,
    "name": "블라우스",
            "supercategory": "상의"
}, {
    "id": 6,
    "name": "브라탑",
            "supercategory": "상의"
}, {
    "id": 7,
    "name": "팬츠",
            "supercategory": "하의"
}, {
    "id": 8,
    "name": "조거팬츠",
            "supercategory": "하의"
}, {
    "id": 9,
    "name": "청바지",
            "supercategory": "하의"
}, {
    "id": 10,
    "name": "래깅스",
            "supercategory": "하의"
}, {
    "id": 11,
    "name": "스커트",
            "supercategory": "하의"
}, {
    "id": 12,
    "name": "패딩",
            "supercategory": "아우터"
}, {
    "id": 13,
    "name": "짚업",
            "supercategory": "아우터"
}, {
    "id": 14,
    "name": "점퍼",
            "supercategory": "아우터"
}, {
    "id": 15,
    "name": "재킷",
            "supercategory": "아우터"
}, {
    "id": 16,
    "name": "베스트",
            "supercategory": "아우터"
}, {
    "id": 17,
    "name": "가디건",
            "supercategory": "아우터"
}, {
    "id": 18,
    "name": "코트",
            "supercategory": "아우터"
}, {
    "id": 19,
    "name": "드레스",
            "supercategory": "드레스"
}, {
    "id": 20,
    "name": "점프수트",
            "supercategory": "드레스"
}, {
    "id": 21,
    "name": "UNKNOWN",
            "supercategory": "UNKNOWN"
}]

ann = []

for i, label_path in enumerate(label_paths):
    json_file = load_json(label_path)
    big_cat = label_path.split('/')[-2]
    data = json_file['데이터셋 정보']['데이터셋 상세설명']['라벨링']
    img_h = json_file['이미지 정보']['이미지 높이']
    img_w = json_file['이미지 정보']['이미지 너비']
    file_name = json_file['이미지 정보']['이미지 파일명']
    file_name = file_name[:-3]+'jpg'
    lst.append({'license': 0, 'file_name': os.path.join(
        '원천데이터', big_cat, file_name), 'height': img_h, 'width': img_w, 'id': i})
    for keyword in ['아우터', '하의', '원피스', '상의']:
        if len(data[keyword][0]) != 0:
            try:
                c = data[keyword][0]['카테고리']
            except KeyError:
                c = 'UNKNOWN'
            cat_ids = cat_list[c]
            bb_info = json_file['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'][keyword][0]
            polygon = json_file['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][keyword][0]
            tmp = []
            try:
                b_x = bb_info['X좌표']
                b_y = bb_info['Y좌표']
                b_w = bb_info['가로']
                b_h = bb_info['세로']
                for i in range(1, len(polygon)//2+1):
                    x = polygon['X좌표'+str(i)]
                    y = polygon['Y좌표'+str(i)]
                    tmp.append(x)
                    tmp.append(y)
                bbox = [b_x, b_y, b_w, b_h]
                ann.append({'id': len(ann), 'image_id': i, 'category_id': cat_ids,
                           'iscrowd': 0, 'segmentation': [tmp], 'area': b_w*b_h, 'bbox': bbox})
            except KeyError:
                continue

file_data['images'] = lst
file_data['categories'] = cat
file_data['annotations'] = ann

with open('train_all.json', 'w', encoding="utf-8") as make_file:
    json.dump(file_data, make_file, ensure_ascii=False, indent="\t")


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def main():
    with open('./train_all.json', 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(
            lambda a: int(a['image_id']), annotations)

        images = funcy.lremove(
            lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=0.8)

        save_coco('./train.json', info, licenses, x,
                  filter_annotations(annotations, x), categories)
        save_coco('./val.json', info, licenses, y,
                  filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(x), './train.json', len(y), './val.json'))
