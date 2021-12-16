"""
"""
import transformers

import torch
from torch import nn
import torchvision.models as models
from models.attention import NewAttention
from models.fc import FCNet


class VQAModel(nn.Module):
    def __init__(self, num_targets, dim_h=1024, large=False, res152=False):
        super(VQAModel, self).__init__()

        # The BERT model: 질문 --> Vector 처리를 위한 XLM-Roberta모델 활용
        self.bert = transformers.XLMRobertaModel.from_pretrained(
            'xlm-roberta-base')

        # Backbone: 이미지 --> Vector 처리를 위해 ResNet50을 활용
        self.q_linear = nn.Linear(768, 1024)
        self.q_relu = nn.ReLU()
        self.q_drop = nn.Dropout(0.2)
        self.v_att = NewAttention(1024, dim_h, dim_h)
        self.q_net = FCNet([1024, 1024])
        self.v_net = FCNet([1024, 1024])
        # classfier: MLP기반의 분류기를 생성
        self.linear = nn.Linear(dim_h, num_targets)

    def forward(self, idx, mask, image):
        # image shape = (b,k,1024)
        batch_size = image.shape[0]
        output = self.bert(idx, mask)
        q_f = output['pooler_output']
        # q_f의 차원 = (b,1024)
        q_f = self.q_drop(self.q_linear(q_f))
        # i_f의 차원 = (b,k,1024)
        i_f = image

        # (b,k,1024)
        att = self.v_att(i_f, q_f)

        # (b,k,1)
        v_emb = (att*i_f).sum(1)

        # (b,1024)
        q_repr = self.q_net(q_f)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        # MLP classfier로 답변 예측
        return self.linear(joint_repr)
