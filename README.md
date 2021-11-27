# Fashion-Reader
![image](https://github.com/Fashion-Reader/fashion-reader/blob/main/etc/img.png)

## Task :computer:
### Online clothing shopping assistant for the visually impaired :speaking_head:

## Members :family_man_woman_girl_boy:

| [권태양](https://github.com/sunnight9507) | [류재희](https://github.com/JaeheeRyu) | [박종헌](https://github.com/PJHgh) | [신찬엽](https://github.com/chanyub) |
[오수지](https://github.com/ohsuz) | [이현규](https://github.com/LeeHyeonKyu) | [정익효](https://github.com/dlrgy22) | [조원](https://github.com/jo-member) | [최유라](https://github.com/Yuuraa) |

## More Info :question:
- [Notion](https://www.notion.so/Fashion-Reader-9244c753b78b470a9355a51478a38e83)
- [PPT](https://github.com/Fashion-Reader/fashion-reader/blob/main/etc/FASHION-READER.pdf)

## Setup

### 1) Data Crawling
```
$ cd 01_Data
$ pip install -r requirements.txt
$ bash run.sh
```

### 2) Training VQA Model
```
$ cd 03_VQA_Model
$ python3 train.py
```

### 3) Training QA model
```
$ cd 04_QA_Model
question_intention_clf.ipynb
```

### 4) Run API Server
```
$ cd 02_RestAPI_server
$ pip install -r requirements.txt
$ bash run.sh
```

### 5) Run APP Server
```
$ cd 05_App
```
