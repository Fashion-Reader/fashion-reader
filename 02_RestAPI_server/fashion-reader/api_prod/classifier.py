# model instance 생성
# model parameter load
# 전달 받은 query를 model에 입력으로 넣고,
# 질문에 해당하는 class index를 받음.
# index랑 class의 key값을 치환
# item id와 index에 해당하는 data return

import random
import torch
from .serializers import ProductsSerializer
from .models import Product
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class QueryModule():
    def __init__(self):
        self.model_name = "monologg/koelectra-small-v3-discriminator" # we need to change this
        self.max_length = 35
        self.classes = {
            0: '소매기장',
            1: '소재',
            2: '색깔',
            3: '프린팅',
            4: '핏',
            5: '카테고리',
            6: '기장',
            7: '넥라인'}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.classes))
        # self._load_pretraind_model()

    def estimation(self, query, item_id):
        tokenized_query = self._tokenizing(query)
        pred = self.model(tokenized_query)[0] # we may change this
        pred = int(torch.argmax(pred, -1))
        return self.classes[pred]

    def _load_pretraind_model(self, model_path='./base.pt'):
        self.model.load_state_dict(torch.load(model_path))

    def _get_item(self, item_id):
        user_serializer = ProductsSerializer(Products.objects.get(item_id=item_id))
        return user_serializer.data

    def _tokenizing(self, text):
        return self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
