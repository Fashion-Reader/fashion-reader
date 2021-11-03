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
         14: '상의 색',
         15: '하의 색',
         16: '상의 카테고리',
         17: '하의 카테고리',
         18: '넥 라인'}

name2column = {
    "이름": ("products", "name"),
    "가격": ("products", "price"),
    "사이즈 옵션": ("products", "size_options"),
    # "소재": (""),
    "비침": ("products", "see_through"),
    "카테고리": ("products", "item_type"),
    "색 옵션": ("products", "color_options"),
    "두께감": ("products", "thickness"),
    "신축성": ("products", "flexibility"),
    "촉감": ("products", "touch"),
    "핏": ("products", "fit"),
    "안감": ("products", "lining"),
    "스타일": ("vqa_table", "style"),
    "프린팅": ("vqa_table", "printing"),
    "상의 색": ("vqa_table", "top_color"),
    "하의 색": ("vqa_table", "bottom_color"),
    "상의 카테고리": ("vqa_table", "top_category"),
    "하의 카테고리": ("vqa_table", "bottom_category"),
    "넥 라인": ("vqa_table", "neckline"),
}

# Load model
network = torch.load("/home/ec2-user/fashion_reader/03_Model/QA_model.pt", map_location=torch.device('cpu'))
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

def load_vqa_answer(item_id, columns):
    # CHECK TABLE
    db, column = name2column[columns]
    SQL = f"SELECT {column} from {db} where item_id = \"{item_id}\""
    cursor.execute(SQL)
    try:
        return cursor.fetchall()[0][0]
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
    return JsonResponse({'query':query, 'response':column + " : " + res}, json_dumps_params = {'ensure_ascii': True})