from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
from .serializers import ProductsSerializer
from rest_framework import status
from .models import Products
import urllib.parse
from .classifier import QueryModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, ElectraForSequenceClassification, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import pymysql
from pymongo import MongoClient

names = {0: '이름',
         1: '가격',
         2: '사이즈 옵션',
         3: '소재',
         4: '비침',
         5: '카테고리',
         6: '색 옵션',
         7: '두께감',
         8: '신축성',
         9: '촉감',
         10: '핏',
         11: '안감',
         12: '스타일',
         13: '프린팅',
         14: '넥 라인'}

name2column = {
    "이름": ("product", "name"),
    "가격": ("product", "price"),
    "사이즈 옵션": ("product", "size_options"),
    # "소재": (""),
    "비침": ("product", "see_through"),
    "카테고리": ("product", "item_type"),
    "색 옵션": ("product", "color_options"),
    "두께감": ("product", "thickness"),
    "신축성": ("product", "flexibility"),
    "촉감": ("product", "touch"),
    "핏": ("product", "fit"),
    "안감": ("product", "lining"),
    "스타일": ("vqa_table", "style"),
    "프린팅": ("vqa_table", "printing"),
    "넥 라인": ("vqa_table", "neckline"),
}

# Load model
network = torch.load("/home/ec2-user/fashion_reader/04_QA_model/QA_model.pt", map_location=torch.device('cpu'))
pretrained_model_state = deepcopy(network.state_dict())
network.load_state_dict(pretrained_model_state)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)
network.eval()

# CONNECT DB
db = pymysql.connect(host='database.cawpd3yaf0pl.us-east-2.rds.amazonaws.com',
                     user='admin',
                     passwd='00000000')
cursor = db.cursor()

# USE DATABASE
SQL = "use products_table"
cursor.execute(SQL)

class CFG:
    tokenizer_max_length = 35
    batch_size = 1
    model_name = "xlm-roberta-base"

class NLP_Dataset_test(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataset = dataframe
        self.question = dataframe['question']
        self.labels = dataframe['label']
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(self.question[idx],
                                        max_length=CFG.tokenizer_max_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',
                                        add_special_tokens=True)
        
        tokenized_text['label'] = self.labels[idx]
        return tokenized_text

    def __len__(self):
        return len(self.labels)


def question_model(query):
    input_str = query
    test_df = pd.DataFrame(zip([input_str],[-1]), columns=['question', 'label'])
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    test_set = NLP_Dataset_test(test_df, tokenizer)
    test_loader = DataLoader(dataset=test_set, batch_size=CFG.batch_size, shuffle=False)

    preds_all = []
    prediction_array=[]
    with tqdm(test_loader, total=test_loader.__len__(), unit='batch') as test_bar:
        for batch in test_bar:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            preds = network(input_ids, attention_mask = attention_mask)[0]
            preds_all += [torch.argmax(preds, 1).detach().cpu().numpy().item()]

    return names[preds_all[0]]


def answer(column, value):
    print(column, value)
    if column in {"이름", "가격", "사이즈 옵션", "카테고리", "색 옵션", "스타일", "프린팅", "넥 라인"}:
        return f"해당 상품의 {column}은 {value} 입니다."
    elif column == "비침":
        if value == "있음":
            return "해당 상품은 비치는 편입니다."
        elif value == "약간있음":
            return "해당 상품은 약간 비치는 편입니다."
        elif value == "없음":
            return "해당 상품은 비치지 않는 편입니다."
        else:
            return "밝은 컬러는 비치는 편입니다."
    elif column == "두께감":
        if value == "두꺼움":
            return "해당 상품은 두꺼운 편입니다."
        elif value == "보통":
            return "해당 상품의 두께는 보통인 편입니다."
        else:
            return "해당 상품은 얇은 편입니다."
    elif column == "신축성":
        if value == "좋음":
            return "해당 상품의 신축성은 좋은 편입니다."
        elif value == "약간 있음":
            return "해당 상품의 신축성은 약간 있는 편입니다."
        else:
            return "해당 상품의 신축성은 없는 편입니다."
    elif column == "촉감":
        if value == "부드러움":
            return "해당 상품의 촉감은 부드러운 편입니다."
        elif value == "보통":
            return "해당 상품의 촉감은 부드러움과 거침의 중간 정도입니다."
        else:
            return "해당 상품의 촉감은 거친 편입니다."
    elif column == "핏":
        if value == "루즈핏":
            return "해당 상품은 루즈핏입니다."
        elif value == "기본핏":
            return "해당 상품은 기본핏입니다."
        else:
            return "해당 상품은 타이트한 핏입니다."
    elif column == "안감":
        if {"전체", "부분", "없음"} in value:
            return "해당 상품의 안감은 평범합니다."
        elif value == "기모":
            return "해당 상품의 안감은 기모입니다."
        else:
            return "해당 상품의 안감은 퍼입니다."
    

def load_vqa_answer(item_id, columns):
    # CHECK TABLE
    db, column = name2column[columns]
    SQL = f"SELECT {column} from {db} where item_id = \"{item_id}\""
    cursor.execute(SQL)
    try:
        return answer(columns, cursor.fetchall()[0][0])
    except:
        return "잘 모르겠어요"


class ProdView(APIView):
    def get(self, request, **kwargs):
        if kwargs.get('item_id'):
            query_id = kwargs.get('item_id')
            user_serializer = ProductsSerializer(Products.objects.get(item_id=query_id))
            return Response(user_serializer.data, status=status.HTTP_200_OK)
        elif kwargs.get('item_type_id'):
            query_id = kwargs.get('item_type_id')
            user_serializer = ProductsSerializer(Products.objects.filter(item_type_id=query_id), many=True)
            return Response(user_serializer.data, status=status.HTTP_200_OK)
        else:
            user_queryset = Products.objects.all()
            user_queryset_serializer = ProductsSerializer(user_queryset, many=True)
            return Response(user_queryset_serializer.data, status=status.HTTP_200_OK)


def query_to_response(request, **kwargs):
    query = kwargs.get("query")
    item_id = kwargs.get("item_id")
    column = question_model(query)
    res = load_vqa_answer(item_id, column)
    print(res)
    return JsonResponse({'query': query, 'response': res}, json_dumps_params = {'ensure_ascii': True})
    