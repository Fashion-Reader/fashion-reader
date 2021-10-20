from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
from .serializers import ProductsSerializer
from rest_framework import status
from .models import Products
import urllib.parse
from .classifier import QueryModule

query_module = QueryModule()


from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def similarity_comparison(query):
    sentences = ["소매기장", "소재", "색깔", "프린팅", "핏", "카테고리", "기장", "넥라인", "스타일"]
    # model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    model = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1")


    query_embedding = model.encode(query)
    embeddings = model.encode(sentences)

    result = {}

    for idx, embedding in enumerate(embeddings):
        print(query, sentences[idx])
        print(cos_sim(query_embedding, embedding))
        print()
        result[sentences[idx]] = cos_sim(query_embedding, embedding)

    return sorted(result.items(), key=lambda x: x[1], reverse=True)[0][0]


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
    query = kwargs.get('query')
    item_id = kwargs.get('item_id')
    # res = query_module.estimation(query, item_id)
    res = similarity_comparison(query)
    print(res)
    return JsonResponse({'query':query, 'response':res}, json_dumps_params = {'ensure_ascii': True})