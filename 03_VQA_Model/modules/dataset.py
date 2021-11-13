"""Dataset 클래스 정의

"""

import torch
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, df, tokenizer):
        super().__init__()
        self.df = df.reset_index(drop=False).copy()
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int):
        labels = self.df[['카테고리_ID','스타일_ID','넥라인_ID','무늬_ID','단추','레이스','지퍼','포켓','슬릿','버클','서브 대분류_ID','서브 카테고리_ID','서브 색상_ID']].iloc[index].values

        language_input = self.df['language_input'].iloc[index]
        tokens = self.tokenizer([language_input], padding='max_length', max_length=50)
        
        input_ids = torch.tensor(tokens["input_ids"]).squeeze()
        attention_mask = torch.tensor(tokens["attention_mask"]).squeeze()
        token_type_ids = torch.tensor(tokens["token_type_ids"]).squeeze()

        # [MASK] = 103 token
        mask_pos_ids = torch.tensor([1 if token == 103 else 0 for token in input_ids])

        img_embed_path = self.df['image_embed_path'].iloc[index]
        visual_embeds = torch.tensor([np.load(img_embed_path)])
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).squeeze()
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).squeeze()

        return (input_ids, attention_mask, token_type_ids, visual_embeds.squeeze(), visual_attention_mask, visual_token_type_ids, mask_pos_ids), labels


if __name__ == '__main__':
    pass

