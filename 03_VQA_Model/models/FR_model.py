"""
"""

import torch

from torch import nn
from torch.nn import functional as F
from transformers import VisualBertModel

class FRModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
        self.hidden_size = 768
        
        # 총 13개의 분류 수행
        self.fc_category = nn.Linear(self.hidden_size, 21)
        self.fc_style = nn.Linear(self.hidden_size, 23)
        self.fc_neckline = nn.Linear(self.hidden_size, 13)
        self.fc_pattern = nn.Linear(self.hidden_size, 16)
        ### 디테일
        self.fc_button = nn.Linear(self.hidden_size, 1)
        self.fc_lace = nn.Linear(self.hidden_size, 1)
        self.fc_zipper = nn.Linear(self.hidden_size, 1)
        self.fc_pocket = nn.Linear(self.hidden_size, 1)
        self.fc_slit = nn.Linear(self.hidden_size, 1)
        self.fc_buckle = nn.Linear(self.hidden_size, 1)
        ### 서브 착장
        self.fc_sub_group = nn.Linear(self.hidden_size, 5)
        self.fc_sub_category = nn.Linear(self.hidden_size, 22)
        self.fc_sub_color = nn.Linear(self.hidden_size, 22)

    def forward(self, inputs):
        input_ids, attention_mask, token_type_ids, visual_embeds, visual_attention_mask, visual_token_type_ids, mask_pos_ids = inputs
        outputs = self.model(input_ids=input_ids.int(),
                            attention_mask=attention_mask.int(),
                            token_type_ids=token_type_ids.int(),
                            visual_embeds=visual_embeds.float(),
                            visual_attention_mask=visual_attention_mask.int(),
                            visual_token_type_ids=visual_token_type_ids.int())

        H_all = outputs['last_hidden_state'][:,:50]
        N, L, H = H_all.size()
        mask_pos_ids = mask_pos_ids.unsqueeze(2).expand(H_all.shape)  # (N, L) -> (N, L, 1) -> (N, L, H)
        H_features = torch.masked_select(H_all, mask_pos_ids.bool(), )  # (N, L, H), (N, L, H) -> (N * 13 * H)
        H_features = H_features.reshape(N, 13, H)  # (N * 13 * H) -> (N, 13, H)
        
        # (N, H) -> (N, label_num of each category)
        L_category = self.fc_category(H_features[:, 0, :])
        L_style = self.fc_style(H_features[:, 1, :])
        L_neckline = self.fc_neckline(H_features[:, 2, :])
        L_pattern = self.fc_pattern(H_features[:, 3, :])
        L_button = self.fc_button(H_features[:, 4, :])
        L_lace = self.fc_lace(H_features[:, 5, :])
        L_zipper = self.fc_zipper(H_features[:, 6, :])
        L_pocket = self.fc_pocket(H_features[:, 7, :])
        L_slit = self.fc_slit(H_features[:, 8, :])
        L_buckle = self.fc_buckle(H_features[:, 9, :])
        L_sub_group = self.fc_sub_group(H_features[:, 10, :])
        L_sub_category = self.fc_sub_category(H_features[:, 11, :])
        L_sub_color = self.fc_sub_color(H_features[:, 12, :])
        
        outputs = {"cate_logits": L_category,
                   "style_logits": L_style,
                   "neckline_logits": L_neckline,
                   "pattern_logits": L_pattern,
                   "button_logits": L_button,
                   "lace_logits": L_lace,
                   "zipper_logits": L_zipper,
                   "pocket_logits": L_pocket,
                   "slit_logits": L_slit,
                   "buckle_logits": L_buckle,
                   "sub_group_logits": L_sub_group,
                   "sub_cate_logits": L_sub_category,
                   "sub_color_logits": L_sub_color,
                   }
        return outputs
